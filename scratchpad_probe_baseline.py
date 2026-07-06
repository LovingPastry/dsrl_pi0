"""Probe base-pi0 (Gaussian-noise) success rate across LIBERO tasks to pick a
harder-but-learnable task for DSRL. Mirrors perform_control_eval's baseline path
(i==0) exactly: query_freq=20, noise ~ N(0, I) shape (1,50,32), is_success = reward==1.
"""
import os, sys, time
import numpy as np
import jax
import pathlib

sys.path.insert(0, '/home/fuyx/lanzc/dsrl_pi0')
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi.training import config as openpi_config
from openpi.policies import policy_config
from openpi.shared import download
from examples.train_utils_sim import obs_to_pi_zero_input


class V:  # minimal stand-in for `variant`
    pass


variant = V()
variant.env = 'libero'

SUITE = os.environ.get('PROBE_SUITE', 'libero_10')
EVAL_EP = int(os.environ.get('PROBE_EP', '5'))
MAX_T = int(os.environ.get('PROBE_MAXT', '520'))
QUERY_FREQ = 20
TASK_IDS = [int(x) for x in os.environ.get('PROBE_TASKS', '0,1,2,3,4,5,6,7,8,9').split(',')]

print(f"[probe] suite={SUITE} ep={EVAL_EP} max_t={MAX_T} query_freq={QUERY_FREQ} tasks={TASK_IDS}", flush=True)
config = openpi_config.get_config("pi0_libero")
ckpt = download.maybe_download("s3://openpi-assets/checkpoints/pi0_libero")
agent_dp = policy_config.create_trained_policy(config, ckpt)
print("[probe] pi0 loaded", flush=True)

suite = benchmark.get_benchmark_dict()[SUITE]()
results = []
for tid in TASK_IDS:
    task = suite.get_task(tid)
    variant.task_description = task.language
    bddl = str(pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    succ = 0
    rng = jax.random.PRNGKey(456)
    t0 = time.time()
    for ep in range(EVAL_EP):
        obs = env.reset()
        reward = 0.0
        actions = None
        for t in range(MAX_T):
            if t % QUERY_FREQ == 0:
                rng, key = jax.random.split(rng)
                obs_pi = obs_to_pi_zero_input(obs, variant)
                noise = jax.random.normal(key, (1, 50, 32))
                actions = agent_dp.infer(obs_pi, noise=noise)["actions"]
            a = actions[t % QUERY_FREQ]
            obs, reward, done, _ = env.step(a)
            if done:
                break
        succ += int(reward == 1)
    env.close()
    dt = time.time() - t0
    rate = succ / EVAL_EP
    results.append((tid, rate, succ, EVAL_EP, task.language))
    print(f"[probe] task {tid}: {succ}/{EVAL_EP} = {rate:.0%}  ({dt:.0f}s)  :: {task.language}", flush=True)

print("\n[probe] ===== SUMMARY (sorted by baseline desc) =====", flush=True)
for tid, rate, succ, ep, lang in sorted(results, key=lambda r: -r[1]):
    print(f"  task {tid:2d}: {rate:5.0%} ({succ}/{ep})  {lang}", flush=True)
print("[probe] DONE", flush=True)
