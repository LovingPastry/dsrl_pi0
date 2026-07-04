from tqdm import tqdm
import numpy as np
from jaxrl2.utils.tensorboard_logger import Video, Image
import jax
from openpi_client import image_tools
import math
import PIL
import os
import pickle

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def build_sac_obs(variant, agent_dp, curr_image, qpos, obs_pi_zero=None):
    '''
    Build the SAC observation dict for one query step.
    obs_mode 'pixels': learned-CNN input (image + qpos), the original DSRL setup.
    obs_mode 'vlm': the frozen pi0 PaliGemma prefix representation (last-token
    pooling, same as the real-robot DSRL path) shared by actor and critic.
    '''
    if variant.get('obs_mode', 'pixels') == 'vlm':
        assert obs_pi_zero is not None
        rep, _ = agent_dp.get_prefix_rep(obs_pi_zero)
        # [:, -1, :] matches the real-robot path (train_utils_real.py). With
        # prompts padded to max_token_len the last position is a PAD slot whose
        # fully-masked attention rows degrade to uniform averaging, so this is
        # effectively a whole-prefix mean-pooled summary — deterministic and
        # observation-dependent, kept for parity with the authors' recipe.
        vlm = np.asarray(jax.numpy.asarray(rep[:, -1, :], dtype=jax.numpy.float32))  # (1, vlm_dim)
        obs_dict = {'vlm': vlm}
    else:
        obs_dict = {'pixels': curr_image[np.newaxis, ..., np.newaxis]}
    if variant.add_states:
        obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
    return obs_dict

def _get_or_build_task_env(task_specs, idx, variant):
    """Return the LIBERO env for task_specs[idx], building it lazily on first use.

    HOST-RAM GUARD: holding all N LIBERO EGL envs alongside a ~14 GB frozen pi0
    OOMs a 31 GB box (pi0 + 10 envs ~= 26 GB). With --max_live_envs K>0 we keep at
    most K envs open (LRU); building a (K+1)-th closes the least-recently-used one.
    Round-robin then rebuilds that task's env (~1-2 s) when it comes round again,
    which is a small fraction of a full rollout. K=0 keeps the old unbounded
    behavior (fine for single-task / few-task / big-RAM machines)."""
    spec = task_specs[idx]
    live = variant.setdefault('_live_env_idx', [])  # LRU order of open task indices
    if spec.get('env') is None:
        max_live = int(variant.get('max_live_envs', 0) or 0)
        if max_live > 0:
            # evict BEFORE building so peak == max_live envs (no transient +1 spike)
            while len(live) >= max_live:
                old = live.pop(0)
                if task_specs[old].get('env') is not None:
                    try:
                        task_specs[old]['env'].close()
                    except Exception as e:
                        print(f'[env pool] close of task idx {old} failed: {e}')
                    task_specs[old]['env'] = None
        from examples.train_sim import _get_libero_env  # deferred to avoid circular import
        env, _ = _get_libero_env(spec['task'], 256, variant.seed)
        spec['env'] = env
        live.append(idx)
    else:
        if idx in live:  # mark most-recently-used
            live.remove(idx); live.append(idx)
    return spec['env']


def _train_state_path(outputdir):
    return os.path.join(outputdir, 'train_state.pkl')


def save_full_state(variant, agent, i, rollout_idx, total_env_steps,
                    online_replay_buffer, offline_replay_buffer):
    """Persist EVERYTHING needed for an exact resume: SAC params (flax), the
    replay buffer(s), step / rollout counters, and the agent + numpy RNG state.
    train_state.pkl is written LAST (atomically) so its presence guarantees the
    flax checkpoint and buffers are already on disk."""
    outputdir = variant.outputdir
    agent.save_checkpoint(outputdir, i, variant.checkpoint_interval)
    online_replay_buffer.save(os.path.join(outputdir, 'online_buffer.pkl'))
    if offline_replay_buffer is not None:
        offline_replay_buffer.save(os.path.join(outputdir, 'offline_buffer.pkl'))
    state = {
        'i': int(i),
        'rollout_idx': int(rollout_idx),
        'total_env_steps': int(total_env_steps),
        'agent_rng': np.asarray(agent._rng),
        'np_rng_state': np.random.get_state(),
    }
    tmp = _train_state_path(outputdir) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(state, f, protocol=4)
    os.replace(tmp, _train_state_path(outputdir))
    print(f'[checkpoint] full state saved at step {i} -> {outputdir}')


def restore_full_state(variant, agent, online_replay_buffer, offline_replay_buffer):
    """Inverse of save_full_state. Returns the state dict, or None (start fresh)
    when no complete checkpoint is present in the output dir."""
    from flax.training import checkpoints as flax_ckpt
    outputdir = variant.outputdir
    sp = _train_state_path(outputdir)
    has_flax = flax_ckpt.latest_checkpoint(outputdir) is not None
    if not (os.path.exists(sp) and has_flax):
        print(f'[resume] no complete checkpoint in {outputdir}; starting fresh.')
        return None
    agent.restore_checkpoint(outputdir)
    online_replay_buffer.restore(os.path.join(outputdir, 'online_buffer.pkl'))
    off_path = os.path.join(outputdir, 'offline_buffer.pkl')
    if offline_replay_buffer is not None and os.path.exists(off_path):
        offline_replay_buffer.restore(off_path)
    with open(sp, 'rb') as f:
        state = pickle.load(f)
    try:
        agent._rng = jax.numpy.asarray(state['agent_rng'])
    except Exception as e:
        print('[resume] could not restore agent RNG:', e)
    try:
        np.random.set_state(state['np_rng_state'])
    except Exception as e:
        print('[resume] could not restore numpy RNG:', e)
    print(f"[resume] restored step={state['i']} rollout_idx={state['rollout_idx']} "
          f"online_buf={len(online_replay_buffer)} from {outputdir}")
    return state


def _run_eval(agent, task_specs, eval_env, i, variant, tb_logger, agent_dp):
    """Dispatch to per-task+mean eval for multi-task, else the single-task eval."""
    if task_specs is not None and len(task_specs) > 1:
        return perform_multitask_eval(agent, task_specs, i, variant, tb_logger, agent_dp)
    return perform_control_eval(agent, eval_env, i, variant, tb_logger, agent_dp)


def perform_multitask_eval(agent, task_specs, i, variant, tb_logger, agent_dp):
    """Evaluate the shared agent on each task; log per-task curves under
    evaluation/success_rate/task_<id> and the task-mean as the headline
    evaluation/success_rate."""
    num_tasks = len(task_specs)
    K = variant.get('tasks_per_eval', -1)
    idxs = list(range(num_tasks))
    if isinstance(K, int) and 0 < K < num_tasks:
        start = (i // max(int(variant.eval_interval), 1)) % num_tasks
        idxs = [(start + j) % num_tasks for j in range(K)]
    srs = []
    for ti in idxs:
        spec = task_specs[ti]
        env_ti = _get_or_build_task_env(task_specs, ti, variant)
        variant.task_description = spec['description']
        variant.cur_task_index = ti
        sr = perform_control_eval(agent, env_ti, i, variant, tb_logger, agent_dp,
                                  task_tag=f"task_{spec['task_id']}")
        srs.append(sr)
    mean_sr = float(np.mean(srs))
    tb_logger.log({'evaluation/success_rate': mean_sr}, step=i)
    tb_logger.log({'evaluation/num_tasks_evaled': len(idxs)}, step=i)
    print(f'[multitask eval] step {i}: mean success over {len(idxs)}/{num_tasks} tasks = {mean_sr:.3f}')
    return mean_sr


def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, tb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None,
                                       offline_replay_buffer=None, task_specs=None):
    if offline_replay_buffer is not None:
        # dual-buffer scheme: 50/50 batches from the frozen warmup buffer and the
        # growing online buffer
        from flax.core import frozen_dict
        from jaxrl2.data.dataset import concat_recursive

        def _mixed_iterator(off_buf, on_buf, batch_size):
            half = batch_size // 2
            while True:
                b_off = off_buf.sample(half)
                b_on = on_buf.sample(batch_size - half)
                yield frozen_dict.freeze(concat_recursive([b_off, b_on]))

        replay_buffer_iterator = _mixed_iterator(offline_replay_buffer, online_replay_buffer, variant.batch_size)
    else:
        replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    warmup_trajs = variant.get('warmup_trajs', -1) or -1

    def _updates_ready():
        if warmup_trajs > 0:
            warmup_buf = offline_replay_buffer if offline_replay_buffer is not None else online_replay_buffer
            ready = warmup_buf._traj_counter >= warmup_trajs
        else:
            ready = len(online_replay_buffer) > variant.start_online_updates
        if offline_replay_buffer is not None:
            # need at least one online traj before mixed sampling can start
            ready = ready and len(online_replay_buffer) > 0
        return ready

    total_env_steps = 0
    i = 0
    rollout_idx = 0
    num_tasks = len(task_specs) if task_specs else 1

    # ---- resume: restore SAC params, replay buffer(s), step/rollout counters, RNG ----
    resumed = False
    if variant.get('resume', 0):
        state = restore_full_state(variant, agent, online_replay_buffer, offline_replay_buffer)
        if state is not None:
            i = int(state['i'])
            rollout_idx = int(state['rollout_idx'])
            total_env_steps = int(state['total_env_steps'])
            resumed = True

    if not resumed:
        tb_logger.log({'num_online_samples': 0}, step=i)
        tb_logger.log({'num_online_trajs': 0}, step=i)
        tb_logger.log({'env_steps': 0}, step=i)

    early_stop = variant.get('early_stop_success', 1)
    converged = False  # early-stop: set True once an eval reaches 100% success
    with tqdm(total=variant.max_steps, initial=i) as pbar:
        while i <= variant.max_steps:
            # round-robin task selection for multi-task training: each task is
            # visited in turn, so the first `num_tasks` Gaussian warmup trajs seed
            # exactly one trajectory per task, and later actor trajs keep cycling.
            if task_specs is not None:
                ti = rollout_idx % num_tasks
                cur_env = _get_or_build_task_env(task_specs, ti, variant)
                variant.task_description = task_specs[ti]['description']
                variant.cur_task_index = ti
            else:
                cur_env = env

            if warmup_trajs > 0:
                # Gaussian noise only while filling the warmup quota; in
                # dual-buffer mode the first online traj already comes from the
                # (untrained) actor rather than an extra Gaussian one.
                warmup_buf = offline_replay_buffer if offline_replay_buffer is not None else online_replay_buffer
                use_gaussian = warmup_buf._traj_counter < warmup_trajs
            else:
                use_gaussian = (i == 0 and not resumed)
            traj = collect_traj(variant, agent, cur_env, i, agent_dp, use_gaussian=use_gaussian)
            if (offline_replay_buffer is not None
                    and offline_replay_buffer._traj_counter < warmup_trajs):
                add_online_data_to_buffer(variant, traj, offline_replay_buffer)
                print('offline (frozen) buffer timesteps length:', len(offline_replay_buffer))
                print('offline (frozen) buffer num traj:', offline_replay_buffer._traj_counter)
            else:
                add_online_data_to_buffer(variant, traj, online_replay_buffer)
            rollout_idx += 1
            traj_id = online_replay_buffer._traj_counter - 1
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)

            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if _updates_ready():
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0 and not resumed:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            init_success_rate = _run_eval(agent, task_specs, eval_env, i, variant, tb_logger, agent_dp)
                            if early_stop and init_success_rate is not None and init_success_rate >= 1.0:
                                print(f'[early-stop] baseline eval reached 100% success at step {i}; stopping.')
                                converged = True
                                break
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, tb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                tb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                tb_logger.log_histogram(f'training/{k}', v, i)
                        # tb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        tb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        tb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        tb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        tb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            eval_success_rate = _run_eval(agent, task_specs, eval_env, i, variant, tb_logger, agent_dp)
                            if early_stop and eval_success_rate is not None and eval_success_rate >= 1.0:
                                print(f'[early-stop] eval reached 100% success at step {i}; stopping.')
                                if variant.checkpoint_interval != -1:
                                    save_full_state(variant, agent, i, rollout_idx, total_env_steps,
                                                    online_replay_buffer, offline_replay_buffer)
                                converged = True
                                break
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, tb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        save_full_state(variant, agent, i, rollout_idx, total_env_steps,
                                        online_replay_buffer, offline_replay_buffer)

                if converged:
                    break

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    env_actions = traj.get('env_actions', None)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)

        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        if env_actions is not None and len(env_actions) == episode_len:
            # denoised executed action chunks; only stored by buffers built with
            # the matching extra_fields (DSRL-NA), ignored otherwise
            insert_dict['env_actions'] = env_actions[t]
            insert_dict['next_env_actions'] = env_actions[t + 1] if t < episode_len - 1 else env_actions[t]
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None, use_gaussian=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    if use_gaussian is None:
        # legacy behavior: Gaussian noise for all trajs collected before the
        # first gradient step
        use_gaussian = (i == 0)

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []
    env_action_list = []  # denoised action chunks actually executed (for DSRL-NA)

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)

        if t % query_frequency == 0:
            assert agent_dp is not None
            qpos = obs_to_qpos(obs, variant)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            obs_dict = build_sac_obs(variant, agent_dp, curr_image, qpos, obs_pi_zero)

            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            if use_gaussian:
                # for initial round of data collection, we sample from standard gaussian noise
                noise = jax.random.normal(key, (1, *agent.action_chunk_shape))
                noise_repeat = jax.numpy.repeat(noise[:, -1:, :], 50 - noise.shape[1], axis=1)
                noise = jax.numpy.concatenate([noise, noise_repeat], axis=1)
                actions_noise = noise[0, :agent.action_chunk_shape[0], :]
            else:
                # sac agent predicts the noise for diffusion model
                actions_noise = agent.sample_actions(obs_dict)
                actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
            
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            action_list.append(actions_noise)
            obs_list.append(obs_dict)
            env_action_list.append(np.asarray(actions[:query_frequency], dtype=np.float32))

        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    obs_pi_zero = obs_to_pi_zero_input(obs, variant)
    obs_dict = build_sac_obs(variant, agent_dp, curr_image, qpos, obs_pi_zero)
    if 'state' not in obs_dict:
        obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'env_actions': env_action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1
    }

def perform_control_eval(agent, env, i, variant, tb_logger, agent_dp=None, task_tag=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            if t % query_frequency == 0:
                rng, key = jax.random.split(rng)
                assert agent_dp is not None

                obs_pi_zero = obs_to_pi_zero_input(obs, variant)


                if i == 0:
                    # for initial evaluation, we sample from standard gaussian noise to evaluate the base policy's performance
                    noise = jax.random.normal(rng, (1, 50, 32))
                else:
                    qpos = obs_to_qpos(obs, variant)
                    obs_dict = build_sac_obs(variant, agent_dp, curr_image, qpos, obs_pi_zero)
                    actions_noise = agent.sample_actions(obs_dict)
                    actions_noise = np.reshape(actions_noise, agent.action_chunk_shape)
                    noise = np.repeat(actions_noise[-1:, :], 50 - actions_noise.shape[0], axis=0)
                    noise = jax.numpy.concatenate([actions_noise, noise], axis=0)[None]
                    
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        # multi-task eval logs one video PER TASK (rollout 0) to avoid an
        # N_tasks x eval_episodes video blow-up; single-task keeps all rollouts.
        if task_tag is None or rollout_id == 0:
            vid_key = f'eval_video/{task_tag}/{rollout_id}' if task_tag else f'eval_video/{rollout_id}'
            video = np.stack(image_list).transpose(0, 3, 1, 2)
            tb_logger.log({vid_key: Video(video, fps=50, format="mp4")}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    sfx = f'/{task_tag}' if task_tag else ''
    summary_str = f'\n[{task_tag or "eval"}] Success rate: {success_rate}\nAverage return: {avg_return}\n\n'
    tb_logger.log({f'evaluation/avg_return{sfx}': avg_return}, step=i)
    # single-task: headline evaluation/success_rate here; multi-task: per-task
    # curve here and the caller (perform_multitask_eval) writes the task-mean.
    tb_logger.log({f'evaluation/success_rate{sfx}': success_rate}, step=i)
    tb_logger.log({f'evaluation/avg_episode_len{sfx}': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        tb_logger.log({f'evaluation/Reward >= {r}{sfx}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)
    return success_rate

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, tb_logger):
    if replay_buffer._traj_counter < 1:
        return
    trajs = replay_buffer.get_random_trajs(3)
    if len(trajs['observations']) == 0 or 'pixels' not in trajs['observations'][0]:
        return  # vlm obs mode stores no images; skip the Q-value strip visualization
    images = agent.make_value_reward_visulization(variant, trajs)
    tb_logger.log({'reward_value_images': Image(images)}, step=i)
  
