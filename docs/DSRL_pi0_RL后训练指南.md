# DSRL × π₀：用隐空间强化学习对预训练 π₀ 做 RL 后训练 —— 详细开发指南

> 论文：*Steering Your Diffusion Policy with Latent Space Reinforcement Learning*（DSRL, CoRL 2025, [arXiv:2506.15799](https://arxiv.org/abs/2506.15799)）
> 本文基于对本仓库源码的逐行核验编写，所有结论均给出 `文件:行号` 出处。配套的要点已写入项目级 memory（见 `memory/dsrl-*.md`）。

---

## 1. 算法核心思想

**一句话**：不去微调 π₀ 的任何权重，而是**冻结 π₀**，再训练一个很小的 SAC 智能体，让它在 π₀ 的 **flow-matching 初始噪声空间**里做强化学习——RL 学的是"该往去噪过程里喂哪个噪声"，而 π₀ 把这个噪声确定性地解码成机器人动作。

### 为什么成立

一个训练好的扩散/流匹配策略，一旦权重冻结，就是一个**确定性函数** `a = g(s, w)`：

- `s` 是观测，`w` 是初始噪声（通常 `w ~ N(0, I)`）；
- 多步去噪 / ODE 积分把 `w` 变成动作 `a` 的过程是**固定的、确定的**；
- 基础策略里唯一的随机性就是 `w` 的采样。

因此「**选择 `w` ≡ 选择动作 `a`**」。让 `w` 遍历其支撑集，`a` 就遍历 π₀ 已经学好的"高质量动作流形"。DSRL 训练一个小策略 `π^W(w|s)` 去挑选噪声，从而**操纵**最终动作分布。

### 为什么 sample-efficient

1. **不需要对扩散模型反向传播**：π₀ 只作为一个前向黑盒解码器在"选动作"时被调用，RL 的梯度从不穿过它。可训练参数只有一个小 MLP（论文 π₀ 场景约 50 万参数）。
2. **探索是"有方向"的**：噪声空间 actor 继承了 BC 策略已经学好的动作流形，从一开始探索就落在合理动作附近，而不是在原始动作空间里乱撞。
3. 相比直接微调扩散权重（需要 backprop-through-denoising，且常不稳定），DSRL 把后训练变成一个**低维、固定维度、黑盒**的小型 RL 问题。

### 与"直接微调权重"的区别

|                      | 直接微调 π₀ 权重 | DSRL（本仓库）                                                                                |
| -------------------- | ------------------ | --------------------------------------------------------------------------------------------- |
| 改动对象             | π₀ 全部/部分权重 | 仅一个小 SAC actor/critic                                                                     |
| 是否反传穿过扩散模型 | 是（昂贵、易崩）   | 否                                                                                            |
| 对基础模型的访问     | 需要白盒+梯度      | **黑盒**，只需能注入初始噪声                                                            |
| 可训练参数量         | 数亿               | ~10⁵                                                                                         |
| 样本效率             | 低                 | 高                                                                                            |
| 关键前提             | 有训练代码         | **API 必须允许外部传入初始噪声 `w`**（多数部署 API 不暴露，这是 DSRL 唯一的硬性前提） |

> 论文里有两种变体：**DSRL-SAC**（直接在噪声空间学单个 `Q^W(s,w)`）和 **DSRL-NA / Noise-Aliasing**（先在原始动作空间学 `Q^A(s,a)` 再蒸馏到噪声空间，更省样本，论文首选）。**本公开仓库实现的是 DSRL-SAC 变体**（直接对噪声动作学 Q），见 [`jaxrl2/agents/pixel_sac/critic_updater.py`](../jaxrl2/agents/pixel_sac/critic_updater.py)。

---

## 2. 系统架构与数据流

每隔 `query_freq` 个环境步，触发一次"查询"：

```
                                   ┌─────────────────────────────────────────────┐
   env obs ─┬─ obs_to_img ───────► │ SAC actor  π^W(w|s)  (小CNN编码器+MLP)        │
            │  (resize 64/128)     │   输出: 单个 32 维噪声向量 w∈[-mag,+mag]^32   │
            │                      └───────────────┬─────────────────────────────┘
            │                                      │  w  (shape (1,32))
            │                                      ▼
            │                       repeat 最后一行填满到 (H,32)，H=50(sim)/10(real)
            │                                      │  noise = (1, H, 32)
            ├─ obs_to_pi_zero_input ──┐            ▼
            │  (224×224, state, prompt)│   ┌──────────────────────────────────────┐
            │                          └──►│  π₀ (冻结)  agent_dp.infer(obs,        │
            │                              │  noise=noise)  → flow ODE 积分          │
            │                              │  x_1=noise(t=1) ──► x_0=动作(t=0)        │
            │                              └───────────────┬──────────────────────┘
            │                                              │ actions: (H, action_dim)
            │                                              ▼
            │                        执行前 query_freq 个动作:  env.step(actions[t%query_freq])
            ▼                                              │
   sparse reward (-1 每步, 0 成功) ◄───────────────────────┘
            │
            ▼
   Replay Buffer  (一次查询 = 一条 transition: obs, w, r=-1/0, next_obs, discount=γ^query_freq)
            │
            ▼
   每个环境步做 multi_grad_step 次 SAC 更新  (UTD=20)
```

**关键点（已逐行核验）**：

- SAC actor 每次只输出**一个 32 维噪声向量**（不是一整段 chunk）。动作空间在 [`examples/train_sim.py:76`](../examples/train_sim.py#L76) 被硬编码成 `Box(shape=(1,32))`，注释明确写着"32 is the noise action space of pi 0"。于是 `action_dim = 32`、`action_chunk_shape = (1,32)`（[`pixel_sac_learner.py:136-137`](../jaxrl2/agents/pixel_sac/pixel_sac_learner.py#L136-L137)）。
- 这个 32 维向量被**重复填满**到 π₀ 的动作 horizon（sim=50，real=10），再作为 `noise` 传给 π₀（[`train_utils_sim.py:240-244`](../examples/train_utils_sim.py#L240-L244)）。所以 π₀ 这一段 50 步动作其实是由同一个 32 维隐变量生成的。
- π₀ 的 `sample_actions(observation, *, noise, num_steps=10)` 把 `noise` 当作 ODE 初值 `x_1`（t=1 为噪声），Euler 积分 10 步到 `x_0`（t=0 为动作）：`x_0,_ = jax.lax.while_loop(cond, step, (noise, 1.0))`（[`openpi/src/openpi/models/pi0.py:268-321`](../openpi/src/openpi/models/pi0.py#L268-L321)）。
- 噪声注入入口：`policy.infer(obs, noise=...)`，若 `noise is None` 则退化为标准高斯（即原版 π₀）——这就是 fork 的改动（[`openpi/src/openpi/policies/policy.py:44-70`](../openpi/src/openpi/policies/policy.py#L44-L70)）。

---

## 3. 代码地图

| 文件                                                                                                                                                                                                                       | 职责                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/scripts/run_{libero,aloha,real}.sh`                                                                                                                                                                            | 启动脚本，设环境变量 + 传 CLI                                                                                                                                                                                                       |
| [`examples/launch_train_sim.py`](../examples/launch_train_sim.py)                                                                                                                                                         | 仿真入口：argparse + 硬编码`train_args_dict`（actor_lr=1e-4, num_qs=10, encoder='small', latent_dim=50…）                                                                                                                        |
| [`examples/train_sim.py`](../examples/train_sim.py)                                                                                                                                                                       | 建环境、`DummyEnv`（定义 obs/动作空间）、**加载 π₀**（`openpi_config.get_config(...)` + s3 checkpoint + `create_trained_policy`）、建 SAC、进训练循环                                                                 |
| [`examples/train_utils_sim.py`](../examples/train_utils_sim.py)                                                                                                                                                           | **核心**：`collect_traj`（rollout + 噪声→动作桥）、`add_online_data_to_buffer`（query 级 transition）、`perform_control_eval`、`obs_to_img/obs_to_pi_zero_input/obs_to_qpos`、`trajwise_alternating_training_loop` |
| [`examples/train_real.py`](../examples/train_real.py) / [`train_utils_real.py`](../examples/train_utils_real.py)                                                                                                         | Franka DROID 真机路径：远程 π₀（websocket）、人工奖励、π₀ VLM 特征进 state                                                                                                                                                      |
| [`jaxrl2/agents/pixel_sac/pixel_sac_learner.py`](../jaxrl2/agents/pixel_sac/pixel_sac_learner.py)                                                                                                                         | `PixelSACLearner`：建 actor/critic/温度网络、`_update_jit`                                                                                                                                                                      |
| [`actor_updater.py`](../jaxrl2/agents/pixel_sac/actor_updater.py) / [`critic_updater.py`](../jaxrl2/agents/pixel_sac/critic_updater.py) / [`temperature_updater.py`](../jaxrl2/agents/pixel_sac/temperature_updater.py) | SAC 三个更新                                                                                                                                                                                                                        |
| [`jaxrl2/networks/learned_std_normal_policy.py`](../jaxrl2/networks/learned_std_normal_policy.py)                                                                                                                         | `LearnedStdTanhNormalPolicy`：噪声 actor，`action_magnitude` 在此约束                                                                                                                                                           |
| [`jaxrl2/networks/values/`](../jaxrl2/networks/values/)                                                                                                                                                                   | `StateActionEnsemble` critic                                                                                                                                                                                                      |
| [`jaxrl2/data/replay_buffer.py`](../jaxrl2/data/replay_buffer.py)                                                                                                                                                         | query 级回放缓冲，容量 =`max_steps // multi_grad_step`                                                                                                                                                                            |
| `openpi/`（子模块，nakamotoo fork，commit `a6d2400`）                                                                                                                                                                  | `models/pi0.py:sample_actions`（可注入噪声的 flow ODE）、`policies/policy.py:infer(noise=)`、新增 `get_prefix_rep`（VLM 特征）                                                                                                |

> 仓库基于 [jaxrl2](https://github.com/ikostrikov/jaxrl2) + [PTR](https://github.com/Asap7772/PTR)。

---

## 4. SAC 学习器细节

全部在 `jaxrl2/agents/pixel_sac/`，是一个 **REDQ 风格的 pixel-SAC**，运行在 **query 级**（一次噪声查询 = 一条 transition）。

### 4.1 动作 / actor

- 动作 = 32 维隐噪声。actor 为 `LearnedStdTanhNormalPolicy`：MLP 输出 mean/log_std → `TanhMultivariateNormalDiag`，先 `Tanh` 压到 (-1,1)，再仿射映射到 `[-action_magnitude, +action_magnitude]`。
- ⚠️ **精确表述**：`action_magnitude` 不是"乘在高斯上的系数"，而是作为对称边界 `low=-action_magnitude, high=action_magnitude` 传入（[`pixel_sac_learner.py:175`](../jaxrl2/agents/pixel_sac/pixel_sac_learner.py#L175)），真正的缩放是 tanh 之后的 affine rescale（[`learned_std_normal_policy.py:48-50`](../jaxrl2/networks/learned_std_normal_policy.py#L48-L50)）。它决定了**被操纵的噪声能偏离 0 多远**（即探索半径 / 偏离基础策略的幅度）。

### 4.2 critic

- `StateActionEnsemble(hidden_dims, num_qs)`，每个 Q 输入 `(编码后的图像观测, 噪声动作)`。
- `critic_reduction='mean'`：对 ensemble 取**均值**（不是 min）。sim `num_qs=10`，real `num_qs=2`。
- 目标：`target_q = r + γ^query_freq · mask · next_q`（[`critic_updater.py:27`](../jaxrl2/agents/pixel_sac/critic_updater.py#L27)），`backup_entropy=False`（目标里不含熵项）。

### 4.3 actor / 温度

- actor loss：`(log_prob · α − Q).mean()`（[`actor_updater.py:53`](../jaxrl2/agents/pixel_sac/actor_updater.py#L53)）。
- 温度 α 自动调到 `target_entropy`：`'auto'` → `-action_dim/2 = -16`；aloha 显式设为 `0.0`。

### 4.4 编码器 / 训练节奏

- actor 和 critic **共享一个小 CNN 编码器**（`encoder_type='small'`，4 层卷积 + spatial-softmax + `latent_dim=50` bottleneck），作用在 resize 后的图像上（[`pixel_sac_learner.py:48-78`](../jaxrl2/agents/pixel_sac/pixel_sac_learner.py#L48-L78)）。**这套编码器与 π₀ 的 VLM 完全独立**（真机路径除外，见 §5）。
- 图像增广：random crop + color jitter。
- UTD = `multi_grad_step`（每个环境步做多少次梯度更新），sim=20 / real=30。
- 当 `len(online_replay_buffer) > start_online_updates` 后才开始更新（[`train_utils_sim.py:118`](../examples/train_utils_sim.py#L118)）。
- 软更新 `tau=0.005`；`actor_lr=1e-4`，`critic_lr=temp_lr=3e-4`。

### 4.5 奖励与 transition（容易看错的地方）

- **奖励**：稀疏、惩罚时间——每个 query 步 `-1`，成功那一步 `0`；失败则全 `-1`。`mask` 仅在成功终止步为 0（[`train_utils_sim.py:277-287`](../examples/train_utils_sim.py#L277-L287)）。即 agent 在最小化"到成功的步数"。
- **transition 粒度**：每次噪声查询存一条，`discount = variant.discount ** query_freq`（半 MDP / 时序扩展动作，[`train_utils_sim.py:165,188`](../examples/train_utils_sim.py#L165)）。

---

## 5. 三套环境配置对比

| 配置项                        | libero                             | aloha_cube                               | franka_droid（真机）             |
| ----------------------------- | ---------------------------------- | ---------------------------------------- | -------------------------------- |
| π₀ config / checkpoint      | `pi0_libero` / s3 `pi0_libero` | `pi0_aloha_sim` / s3 `pi0_aloha_sim` | DROID 模型，**远程 serve** |
| π₀ 动作 horizon             | 50                                 | 50                                       | 10                               |
| query_freq                    | 20                                 | 50                                       | 10                               |
| action_magnitude              | 1.0                                | 2.0                                      | 2.5                              |
| multi_grad_step (UTD)         | 20                                 | 20                                       | 30                               |
| start_online_updates          | 500                                | 1000                                     | —                               |
| resize_image (SAC 编码器输入) | 64                                 | 64                                       | 128                              |
| max_steps                     | 500k                               | 3M                                       | 500k                             |
| discount                      | 0.999                              | 0.999                                    | 0.99                             |
| target_entropy                | auto(=−16)                        | 0.0                                      | auto                             |
| num_qs                        | 10                                 | 10                                       | 2                                |
| hidden_dims                   | 128                                | 128                                      | 1024                             |
| MUJOCO                        | 3.3.1                              | 2.3.7                                    | —                               |

**差异原因**：任务越长 / 越难，给的 `action_magnitude`、`query_freq` 越大（更大的操纵权限、更长的动作块）；真机用更大的编码器输入（128）和更大的 MLP（1024），但更小的 Q 集成（num_qs=2）。

**真机路径的三个关键不同**（[`train_utils_real.py`](../examples/train_utils_real.py)）：

1. **远程 π₀**：`agent_dp = WebsocketClientPolicy(host=remote_host, port=remote_port)`（[`train_real.py:89-91`](../examples/train_real.py#L89-L91)）。先在服务器跑 `cd openpi && python scripts/serve_policy.py --env=DROID`。
2. **SAC 的 state 里加入 π₀ 的 VLM 特征**（fork 新增 `get_prefix_rep` 的用途）：`img_rep_pi0 = agent_dp.get_prefix_rep(request_data)[:, -1, :]`（约 2048 维），与 joint/gripper 位置拼成 SAC 的 `state`（[`train_utils_real.py:158-166`](../examples/train_utils_real.py#L158-L166)）。仿真路径没有这一步。
3. **人工奖励 + 人在回路**：每条 trial 结束由人按 `1`（成功）/`0`（失败）打标，再转成稀疏 -1/0 奖励（[`train_utils_real.py:207-256`](../examples/train_utils_real.py#L207-L256)）；`q`=停止 rollout，`c`=手动 reset 后确认。夹爪动作在 0.5 处二值化，动作 clip 到 [-1,1]。

---

## 6. 从零跑通

### 6.1 安装

```bash
conda create -n dsrl_pi0 python=3.11.11 && conda activate dsrl_pi0
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"
pip install -e openpi && pip install -e openpi/packages/openpi-client
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu   # LIBERO 与 TensorBoard 日志 (torch.utils.tensorboard) 均需要
```

> 子模块：`LIBERO@87edbd1`、`openpi@a6d2400`（均为 nakamotoo fork）。国内拉取 GitHub 走本地 clash 代理 `127.0.0.1:7897`、用 HTTPS。

### 6.2 仿真训练

```bash
bash examples/scripts/run_libero.sh     # 或 run_aloha.sh
```

- 脚本会设 `MUJOCO_GL=egl`、`OPENPI_DATA_HOME=./openpi`（π₀ checkpoint 缓存位置）、`EXP=./logs/<proj>`、`CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_PREALLOCATE=false`。
- 首次运行会从 `s3://openpi-assets/checkpoints/...` 自动下载 π₀ 权重（`download.maybe_download`）。
- `i==0` 时先用**标准高斯噪声**评估一遍基础 π₀，作为 baseline（[`train_utils_sim.py:343-345`](../examples/train_utils_sim.py#L343)）。
- LIBERO 默认任务硬编码：`libero_90`、`task_id=57`、`max_timesteps=400`（[`train_sim.py:111-120`](../examples/train_sim.py#L111-L120)）。
- 日志改用 **TensorBoard**：训练指标写入运行目录 `$EXP/<run_name>`；评估视频（`videos/`，mp4）与 value/reward 可视化（`images/`，png）保存在同一运行目录下。查看：`tensorboard --logdir $EXP`。

### 6.3 真机训练（Franka DROID）

```bash
# [远程服务器] 起 π₀ 服务
cd openpi && python scripts/serve_policy.py --env=DROID
# [机器人客户端] 填好 run_real.sh 里的 remote_host/remote_port + 三个相机 ID，然后
bash examples/scripts/run_real.sh
```

需先装好 DROID 包并配好 Franka。

---

## 7. 超参速查表

| flag                                                           | 含义                                                              | 默认/推荐                        | 调参方向                              |
| -------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------- | ------------------------------------- |
| `--action_magnitude`                                         | 噪声盒子边界`[-m,m]`，= 探索半径 / 偏离基础策略幅度             | 1.0–2.5（论文≈1.5）            | 任务难/需要更大改动→调大；不稳→调小 |
| `--query_freq`                                               | 一次噪声查询控制多少环境步；同时是 discount 指数`γ^query_freq` | 10–50                           | 动作块越长越大；越细粒度控制越小      |
| `--multi_grad_step`                                          | UTD，每环境步更新次数                                             | 20–30                           | 想更省样本→调大（更吃算力）          |
| `--num_qs`                                                   | Q 集成头数（mean 聚合）                                           | sim 10 / real 2                  | 价值估计不稳→调大                    |
| `--target_entropy`                                           | SAC 目标熵；`auto`=−action_dim/2                               | auto / 0.0                       | 想更确定性→调大（如 0.0）            |
| `--discount`                                                 | 基础折扣（再`**query_freq`）                                    | 0.99–0.999                      | 长任务调大                            |
| `--start_online_updates`                                     | 缓冲多少条后开始更新                                              | 500–1000                        | 先攒够数据再学                        |
| `--batch_size`                                               | SAC mini-batch                                                    | 256                              | —                                    |
| `--hidden_dims`                                              | actor/critic MLP 宽度（单值会扩成 3 层）                          | sim 128 / real 1024（论文 2048） | 容量不够→调大                        |
| `--resize_image`                                             | SAC 编码器输入分辨率                                              | 64 / 128                         | 真机/细节多→调大                     |
| `--max_steps`                                                | 总训练步                                                          | 500k–3M                         | —                                    |
| `--eval_interval` / `--log_interval` / `--eval_episodes` | 评估/日志频率                                                     | —                               | —                                    |
| `--env` / `--algorithm`                                    | 环境 / 算法（`pixel_sac`）                                      | libero / pixel_sac               | —                                    |

> 注意：论文推荐的 actor/critic 是 **width-2048 的 3 层 MLP**、`action_magnitude≈1.5`、UTD≈20、10-Q 集成；本仓库 sim 脚本用了更小的 `hidden_dims=128`。要复现论文级效果可适当加大网络。

---

## 8. 如何迁移到「新环境 / 新任务 / 新基础模型」

### 8.1 新仿真环境 / 新任务

1. **建环境**：在 [`examples/train_sim.py:main`](../examples/train_sim.py#L79) 加 `elif variant.env == ...` 分支，设 `env_max_reward`、`max_timesteps`、`task_description`。
2. **三个观测构造函数**（`examples/train_utils_sim.py`）：
   - `obs_to_img`：SAC 编码器看的图（resize）。
   - `obs_to_pi_zero_input`：π₀ 期望的 dict（图像 key、state、prompt），必须与该 π₀ config 的 input transform 对齐。
   - `obs_to_qpos`：SAC 用的低维 state。
3. **π₀ checkpoint/config**：设 `openpi_config.get_config("<name>")` + 对应 s3 checkpoint；`<name>` 必须在 openpi 的 `training/config.py` 里存在。
4. **奖励/成功判定**：`collect_traj` 用 `reward == env_max_reward` 做稀疏 -1/0，替换成你的成功检测器。
5. **动作维度**：`DummyEnv` 动作空间硬编码 `(1,32)`，`32` 是 π₀ padded 动作维。若基础模型动作维不同，要同时改 `DummyEnv` 与 `collect_traj/perform_control_eval` 里 `50`/`32` 这些硬编码（噪声 reshape/repeat 成 `(1, horizon, action_dim)`）。

### 8.2 换一个新的扩散/流匹配基础模型（替换 π₀）

唯一硬性要求：该模型要提供一个 `.infer(obs, noise=...)` 式接口，使策略是**可注入初始噪声**的**确定性**函数。照搬 openpi fork 的改法即可：

- 把外部 `noise` 透传进采样器，并用它当 ODE/去噪的初值（参考 [`pi0.py:268-321`](../openpi/src/openpi/models/pi0.py#L268-L321) 与 [`policy.py:44-70`](../openpi/src/openpi/policies/policy.py#L44-L70)）；
- 对齐噪声张量形状 `(batch, horizon, action_dim)` 与"重复填满 horizon"的逻辑。

只要满足这一点，本仓库的 SAC 学习器几乎可以原样复用——它对基础模型完全黑盒。

### 8.3 可选：升级到论文的 DSRL-NA 变体

本仓库是 DSRL-SAC（直接学 `Q^W(s,w)`）。若要更强的样本效率/离线数据利用，可按论文 Algorithm 1 把 critic 改成"先在动作空间学 `Q^A(s,a)`、再蒸馏到噪声空间 `Q^W(s,w)≈Q^A(s,g(s,w))`"。这需要改 `critic_updater.py` 的目标构造与一次额外的 π₀ 解码。

---

## 9. 常见坑（gotchas）

1. **`query_freq` ≠ RL 动作维**。一个很自然但错误的假设（连对抗校验都专门验过并**否定**了）：RL 动作维不是从 `query_freq` 推出来的，而是硬编码 `action_dim=32`。`query_freq` 只决定**执行节奏**和**折扣指数** `γ^query_freq`。
2. **整段 50 步动作来自同一个 32 维噪声**：actor 只出一个向量，靠 repeat 填满 horizon——这是该实现的简化（"single-latent"），不是 per-step 噪声。
3. **`action_magnitude` 是边界不是乘子**：tanh 之后 affine 到 `[-m,m]`，别理解成"高斯标准差 × m"。
4. **奖励是 -1/0 的时间惩罚式稀疏奖励**，不是 0/1；mask 只在成功终止步为 0。
5. **transition 是 query 级**，回放缓冲容量 `= max_steps // multi_grad_step`，折扣已经是 `γ^query_freq`。
6. **真机 state 里混入了 π₀ 的 VLM 特征**（2048 维），仿真没有——迁移代码时别照抄错路径。
7. **π₀ 始终冻结**：`agent_dp` 是加载好的 trained policy，只前向 `.infer`，梯度从不进它。
8. **openpi 是 fork 的单次 squash commit `a6d2400`**，无法直接 `git diff` 上游；改动集中在 `sample_actions/infer` 的 noise 注入与 `get_prefix_rep`。
9. **MUJOCO 版本**：libero 用 3.3.1，aloha 用 2.3.7，脚本里 `pip install` 切换，别混。
10. **`critic_reduction` 用的是 mean over 10 Qs（REDQ 风格），不是 min**——这是 SAC 默认 min 之外的有意选择。

---

## 10. 与论文结论的对应

- **样本效率来源**：冻结黑盒基础策略 + 噪声空间小 MLP + 不反传扩散模型 + 探索从 BC 流形出发——与 §1 一致。
- **算法**：SAC；actor 输出噪声、critic 评估 `(s, 噪声/动作)`。本仓库 = DSRL-SAC；论文首选 DSRL-NA（噪声混叠 + 动作空间 Q 蒸馏，更省样本、天然吃离线数据）。
- **动作分块**：与 π₀ 一致采用 action chunking（本实现一次解码 50/10 步，执行 query_freq 步）。
- **噪声先验**：`w ~ N(0,I)`，actor 重塑这个初始分布；本实现把噪声盒子约束在 `[-m,m]`。
- **实证范围**：在线 RL（Gym、Robomimic）、离线 RL（10 个 OGBench 任务）、offline-to-online（Robomimic）、generalist 操纵（π₀ on LIBERO、仿真双臂 ALOHA）、真机（Franka / WidowX：pick-and-place、开抽屉、堆叠等）。结论是"用冻结黑盒基础策略达到 SOTA 样本效率"。
- **限制**：需要 API 暴露并能控制初始噪声 `w`（多数部署不暴露）；这是 DSRL 落地的主要前提。
  > 具体每任务成功率请以论文图表为准；网络二手摘要里混入了无关论文的数字，谨慎对待非论文原文的百分比。
  >

---

### 附：本指南的核验方式

本文每条结论均来自对源码的逐行核验，并经一个多智能体工作流（7 个子系统精读 + 5 条核心论断的对抗式独立校验 + 论文 grounding）交叉验证。配套要点已落到项目 memory：`dsrl-core-mechanism`、`dsrl-rl-formulation`、`dsrl-code-map`、`dsrl-env-configs`、`dsrl-adaptation-points`、`dsrl-real-robot-path`。