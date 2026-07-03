# ALOHA sim — one-click 8-condition DSRL ablation

Run the same 8-condition DSRL ablation that we run on LIBERO, but on the
**ALOHA cube-transfer** sim (`gym-aloha`, bimanual, MuJoCo), on a fresh server.

The frozen base policy is **`pi0_aloha_sim`** (π0 finetuned on
`lerobot/aloha_sim_transfer_cube_human`, prompt *"Transfer cube"*), downloaded
automatically from the public openpi S3 bucket.

---

## 1. One-click setup

```bash
git clone <this-repo-url> dsrl_pi0 && cd dsrl_pi0
bash examples/scripts/setup_env.sh
```

`setup_env.sh` does all of:
- `git submodule update --init` for **openpi** and **LIBERO** (falls back to HTTPS if SSH to GitHub is unavailable);
- creates conda env `dsrl_pi0` (python 3.11) and installs `requirements.txt`;
- installs GPU JAX (`jax[cuda12]==0.5.0`) and **`mujoco==2.3.7`** (the version gym-aloha's EGL renderer needs);
- installs the editable packages `openpi`, `openpi-client`, `LIBERO`;
- pre-downloads the **`pi0_aloha_sim`** checkpoint (~12 GB) into `OPENPI_DATA_HOME` (default `~/.cache/openpi`).

Override anything via env vars:
```bash
ENV_NAME=dsrl_pi0 PY_VER=3.11 \
CKPTS="pi0_aloha_sim pi0_libero" \
OPENPI_DATA_HOME=/big/disk/openpi_data_home \
bash examples/scripts/setup_env.sh
```

### GPU note (important)
This repo is pinned to `jax==0.5.0`; the cuDNN/CUDA runtime must match your GPU
driver. Validated with **CUDA 12.8 + cuDNN 9.10.2 on driver 575**. If JAX fails
to initialise cuDNN, install a matching CUDA/cuDNN set — do **not** change the
jax version. Assets are served from a public, anonymous S3 bucket
(`s3://openpi-assets`, us-west-2); no AWS credentials needed.

---

## 2. Run the 8 experiments

```bash
# smoke test first (a few hundred steps, proves the whole pipeline works):
bash examples/scripts/run_ablation8_aloha.sh smoke

# full run: all 8 conditions, sequential, 250k steps each, no early-stop:
bash examples/scripts/run_ablation8_aloha.sh full

# or a subset:
bash examples/scripts/run_ablation8_aloha.sh full baseline vlm na
```

Overridable env (defaults shown):
```bash
MAX_STEPS=250000 EVAL_INTERVAL=12500 CKPT_INTERVAL=50000 \
DEVICE=0 OPENPI_DATA_HOME=~/.cache/openpi EXP_ROOT=~/dsrl_runtime \
bash examples/scripts/run_ablation8_aloha.sh full
```
- `CKPT_INTERVAL=50000` saves the SAC agent every 50k steps (crash recovery / future resume); `-1` disables.
- Runs are **sequential** (single GPU lane, ~10 GB each). Do not parallelise unless you have the VRAM headroom.

### The 8 conditions

| name | improvement 1<br>shared VLM enc | improvement 2<br>replay buffer | improvement 3<br>DSRL-NA | flags |
|------|:---:|:---:|:---:|------|
| `baseline`     | – | original (single buffer) | – | *(none)* |
| `vlm`          | ✔ | original | – | `--obs_mode vlm` |
| `buf1`         | – | 10 warmup trajs, single | – | `--warmup_trajs 10` |
| `buf2`         | – | 10 frozen + online (50/50) | – | `--warmup_trajs 10 --dual_buffer 1` |
| `na`           | – | original | ✔ | `--algorithm pixel_sac_na` |
| `vlm_buf2`     | ✔ | dual | – | `--obs_mode vlm --warmup_trajs 10 --dual_buffer 1` |
| `na_buf2`      | – | dual | ✔ | `--algorithm pixel_sac_na --warmup_trajs 10 --dual_buffer 1` |
| `vlm_na_buf2`  | ✔ | dual | ✔ | all three |

---

## 2b. Parallel run on a multi-GPU server

`run_ablation8_aloha_parallel.sh` (ALOHA) and `run_ablation8_parallel.sh`
(LIBERO) schedule the 8 experiments across the GPUs you list — **one experiment
per GPU**, and if there are fewer GPUs than experiments they queue in a pool
(each GPU grabs the next pending experiment the moment it frees).

```bash
# ALOHA, 4 usable cards — 4 run at once, the other 4 queue:
GPUS=0,1,2,3 bash examples/scripts/run_ablation8_aloha_parallel.sh full

# only cards 2 and 5 are free — all 8 still run, 2-at-a-time:
GPUS=2,5 bash examples/scripts/run_ablation8_aloha_parallel.sh full

# LIBERO parallel (same scheduler, ENV_KIND=libero is the default):
GPUS=0,1,2,3 bash examples/scripts/run_ablation8_parallel.sh full

# validate the scheduling on ANY machine — no GPU, no training (mock jobs):
DRY_RUN=1 GPUS=0,1,2 bash examples/scripts/run_ablation8_parallel.sh full
```

- `GPUS` = comma-separated list of **usable** card IDs; arbitrary IDs are fine
  (e.g. `2,5,7`). Defaults to `0`. Give fewer IDs than experiments and the 8 just
  queue with lower concurrency — nothing is dropped.
- Each job pins both `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` to its card
  (compute + EGL render on the same GPU).
- Same overridable budgets as the sequential runners (`MAX_STEPS`,
  `EVAL_INTERVAL`, `CKPT_INTERVAL`, `EXP_ROOT`, `SUITE`/`TASK_ID` for LIBERO).
- **⚠ Host RAM:** every parallel job loads its own frozen pi0 (**~14 GB system
  RAM**). N-way concurrency needs ~N×14 GB RAM (plus ~10 GB VRAM per card). If the
  box is RAM-limited, list fewer GPUs — the pool still runs all 8, just less
  concurrently. (We OOM-killed a run this way on a 31 GB box.)

---

## 3. View results

```bash
tensorboard --logdir "$HOME/dsrl_runtime/logs/DSRL_pi0_Aloha_ABL8"   # or your $EXP_ROOT
```
- **SCALARS** → `evaluation/success_rate` gives the 8 curves (one per run `exp8_<name>`).
- **IMAGES** → `eval_video/*` shows the rollout videos, embedded per eval step (mp4 files are also saved under each run's `videos/`).

---

## ALOHA-specific settings (baked into the runner)

Differs from the LIBERO runner because ALOHA is a different sim/checkpoint:

| setting | ALOHA | LIBERO | why |
|---|---|---|---|
| `--env` | `aloha_cube` | `libero` | different sim |
| checkpoint | `pi0_aloha_sim` | `pi0_libero` | auto-selected by `--env` in `train_sim.py` |
| `--query_freq` | 50 | 20 | ALOHA action chunk / semi-MDP query cadence |
| `--action_magnitude` | 2.0 | 1.0 | wider noise box for the ALOHA latent |
| `--target_entropy` | 0.0 | auto | matches `run_aloha.sh` |
| env reward | 0..4 (4 = cube transferred) | 0/1 | multi-stage vs binary success |
| SAC state dim | 14 (`agent_pos`) | 8 (eef pose+gripper) | bimanual proprio |
| NA env-action dim | 14 | 7 | per-step action width |

The three algorithm improvements are env-agnostic: the shared-VLM obs uses the
same `get_prefix_rep` last-token pooling (2048-d PaliGemma prefix), the dual
buffer and DSRL-NA learner adapt their action-chunk shapes to ALOHA's dims
automatically.
