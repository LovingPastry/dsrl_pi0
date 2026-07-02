from tqdm import tqdm
import numpy as np
from jaxrl2.utils.tensorboard_logger import Video, Image
import jax
from openpi_client import image_tools
import math
import PIL

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

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, tb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None,
                                       offline_replay_buffer=None):
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
    tb_logger.log({'num_online_samples': 0}, step=i)
    tb_logger.log({'num_online_trajs': 0}, step=i)
    tb_logger.log({'env_steps': 0}, step=i)

    early_stop = variant.get('early_stop_success', 1)
    converged = False  # early-stop: set True once an eval reaches 100% success
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            if warmup_trajs > 0:
                # Gaussian noise only while filling the warmup quota; in
                # dual-buffer mode the first online traj (11th) already comes
                # from the (untrained) actor rather than an extra Gaussian one.
                warmup_buf = offline_replay_buffer if offline_replay_buffer is not None else online_replay_buffer
                use_gaussian = warmup_buf._traj_counter < warmup_trajs
            else:
                use_gaussian = (i == 0)
            traj = collect_traj(variant, agent, env, i, agent_dp, use_gaussian=use_gaussian)
            if (offline_replay_buffer is not None
                    and offline_replay_buffer._traj_counter < warmup_trajs):
                add_online_data_to_buffer(variant, traj, offline_replay_buffer)
                print('offline (frozen) buffer timesteps length:', len(offline_replay_buffer))
                print('offline (frozen) buffer num traj:', offline_replay_buffer._traj_counter)
            else:
                add_online_data_to_buffer(variant, traj, online_replay_buffer)
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
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            init_success_rate = perform_control_eval(agent, eval_env, i, variant, tb_logger, agent_dp)
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
                            eval_success_rate = perform_control_eval(agent, eval_env, i, variant, tb_logger, agent_dp)
                            if early_stop and eval_success_rate is not None and eval_success_rate >= 1.0:
                                print(f'[early-stop] eval reached 100% success at step {i}; stopping.')
                                if variant.checkpoint_interval != -1:
                                    agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)
                                converged = True
                                break
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, tb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

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

def perform_control_eval(agent, env, i, variant, tb_logger, agent_dp=None):
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
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        tb_logger.log({f'eval_video/{rollout_id}': Video(video, fps=50, format="mp4")}, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    tb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    tb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    tb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        tb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
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
  
