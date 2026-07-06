# DSRL-pi0 · 更难评测的两个方向 — 调研与实现计划

> 用 9-agent ultracode 工作流调研（6 调查 + 2 综合 + 1 红队），并对关键事实做了人工二次核实。
> 结论已修正综合器的 1 处事实错误与若干未验证假设。

---

## 0. 一个必须先讲的关键洞察（改变两个方向的前提）

用户的判断是"单任务 LIBERO 太简单 → 分不出方法差异"。**只对了一半。**

红队核实：之前那次 8-condition 消融跑的是 **LIBERO-90 task 46（OOD，base 38%）**，方法们最终收敛到 **0.84–0.98**。也就是说——**即使有 38%→90% 的充足 headroom，8 个方法在收敛点仍然挤在一起。** 但它们的**早期（12.5–37.5k）差异极大（0.40 ~ 0.97）**。

推论：**方法差异主要体现在"学得多快 / 样本效率"，在收敛点被任务天花板压平。** 所以"换更难的任务"这一招要真正生效，必须同时满足两点之一：
1. **任务难到 250k 步都不饱和**（长程/精细正好如此）——差异在终点仍可见；
2. **换评测指标**：不看"最终成功率"，而看 **样本效率 / 曲线下面积(AUC) / 固定预算下（如 50k 步）的成功率**——差异在饱和前就被记录。

> **建议：无论走哪个方向，都把 headline 指标从"末段成功率"改成"到达 X% 所需步数 + AUC"，并在饱和前的固定步数处报一个横切。** 这比单纯堆难度更能稳定区分方法。

---

## 已核实的地面事实（feasibility 的地基）

| 事项 | 事实 | 来源 |
|---|---|---|
| 可用冻结 pi0（sim） | **只有 `pi0_libero` 与 `pi0_aloha_sim`**（均已下载在 `OPENPI_DATA_HOME`） | config.py:483/618；train_sim.py:157-161 |
| config.py 全部 checkpoint | pi0_aloha / _towel / _tupperware / _droid / _fast_droid / _libero / _libero_low_mem / _fast_libero(_low_mem) / _aloha_pen_uncap / **_aloha_sim** / debug — **无 insertion、无 robocasa** | config.py:420-643 |
| checkpoint 下载源 | `s3://openpi-assets/checkpoints/<name>`（匿名公开桶） | config.py:503 等 |
| **GPU** | **RTX 3090 24 GB**（当前已用 5 GB，剩 ~19 GB） | nvidia-smi |
| 主机 RAM | 31 GB 总 / **15 GB 可用**，单 pi0 进程 ~14 GB | free -g |
| pi0 全量微调 | **需 ~70 GB 显存 → 本机不可行** | openpi README |
| pi0 LoRA 微调 | ~22.5 GB（openpi 文档，4090 例）→ **24 GB 上勉强、当前有占用则塞不下** | openpi README |
| 两个 openpi | 推理用 submodule `dsrl_pi0/openpi`；训练另有 `/home/fuyx/lanzc/openpi`（独立 .venv）→ **新 config 要同步进两处** | 磁盘核实 |
| 环境抽象 | 不是类接口，而是 2 个文件里的 `if env=='libero'/elif 'aloha_cube'` 硬分支；jaxrl2/（SAC/buffer）完全 env-无关 | train_sim.py + train_utils_sim.py |

---

# 方向一：LIBERO 多任务训练 + 评测

**目标**：一个共享 SAC agent 同时覆盖一个 suite 内 N 个任务，报 per-task + 均值曲线，让 8-condition 消融在更有 headroom 的多任务设置上把方法拉开。

## 可行性：★★★★☆ 低风险、无需新 checkpoint、向后字节级兼容

整条管线本就 task-agnostic：SAC 输出 32 维 latent noise 播种冻结共享 pi0；任务只经 language prompt 进 pi0，`obs_mode='vlm'` 时经 PaliGemma prefix 进 SAC 状态。单任务耦合只有 3 处（train_sim.py:115-130 建一个 env / 单一 task_description / 单一 success_rate 标量）。ReplayBuffer 是 flat dict + uniform sample，跨任务 shape 一致 → **天然混任务，零 schema 改动**。

## 逐文件改动（~11 处，2-3 文件 + 1 脚本；网络/buffer/TB logger 零改）

1. **`examples/launch_train_sim.py`**（第 34 行后）：新增 4 个顶层 flag `--task_ids / --multitask / --tasks_per_eval / --task_id_onehot`。经 `parse_training_args`(launch_util.py:15-16) 进 `variant` 而**不进 train_kwargs**，不污染 SAC 构造器。
2. **`examples/train_sim.py`**：
   - :115-130 单任务块 → 按 `--task_ids` 建 `task_specs=[{task_id, env, description}]`；`env=eval_env=task_specs[0].env`；`max_timesteps` 仍 per-suite。
   - :73-78 DummyEnv：libero 分支 `state_dim = 8 + task_id_dim`。
   - :200-201 调用处传 `task_specs=`。
   - aloha 分支 `assert multitask==0`。
3. **`examples/train_utils_sim.py`**：
   - 训练循环加 `rollout_idx`，`collect_traj` 前 round-robin `spec = task_specs[rollout_idx % N]`，设 `variant.task_description / cur_task_index`，用 `spec.env` 采集。
   - `build_sac_obs` / collect_traj 末尾 obs：可选把 N 维 task-id one-hot 拼进 `'state'`（经 `_flatten_dict` 进 MLP，能同穿 FeatureMultiplexer 与 PixelMultiplexer）。
   - `perform_control_eval` 加 `task_tag`，log key 加 `/task_{id}` 后缀，video 仅每任务 rollout 0。
   - 新增 `perform_multitask_eval`：逐任务评 → 写 `evaluation/success_rate/task_{id}` + headline `evaluation/success_rate = 均值`。
   - early-stop 改基于均值（消融本就 `--early_stop_success 0`）。
4. **`examples/scripts/run_ablation8_multitask.sh`**（新建）：复制 run_ablation8.sh，`SUITE=libero_10`，加 `--multitask 1 --task_ids 0..9 --task_id_onehot 10 --tasks_per_eval K --early_stop_success 0`，8 条件 variant_flags 不动。

## 任务集选择：libero_10（LIBERO-LONG）全 10 任务（ids 0-9，horizon 520）

- **唯一 in-distribution 且有 headroom 的 suite**：pi0_libero 训练集只含 spatial/object/goal/libero_10 这 4 个 suite。spatial/object/goal 全在 96-99% 天花板（分不开）；libero_10 base 均值 ~85%（长程多阶段半途失败），有真实分散度。
- **算力紧张退化**：4-5 任务子集（先跑 probe 定）。

## ⚠ 修正后的关键风险（红队核实）

1. **【科学性，最重要】premise 不稳**：libero_10 有 **6/10 任务 base≈100%**（在天花板），且单任务消融**即便在有 headroom 的 OOD task46 上也没分开方法**。→ **10 任务均值不保证能区分 8 方法。**
   - **对策**：(a) 把"per-task base 率 probe"从开放问题**提升为 Stage-0 硬门**，按实测 headroom 选任务；(b) headline 用 **hard-subset 均值**（只统计非天花板任务），别用全 10 任务均值；(c) 强烈建议叠加 §0 的**样本效率/AUC 指标**；(d) 先跑 **2-3 任务 pilot 作 go/no-go**，确认方法真能分开再投多日算力。
2. **eval 指标 confound**：`tasks_per_eval K<N` 时 headline 均值每次评的是**不同 K 任务子集**（`start=(i//eval_interval)%N`），曲线会随子集难度锯齿抖动，破坏 8 条件叠图。→ **headline 要么每次全 N 评（用更少 eval_episodes 降本），要么固定一个代表子集**，只在 per-task 下钻里轮转。
3. **RAM/EGL**：`task_specs` 一次性 eager 建 10 个 LIBERO OffScreenRenderEnv（10 个并发 MuJoCo/EGL 上下文）叠在 14 GB pi0 上；本机曾在 31 GB 单 env 下被 OOM 杀两次。→ **Stage-2 smoke 先量峰值 RSS**；逼近 25 GB 就改**惰性建 env**（逐任务 reset），并验证 10 个 EGL 上下文交错 reset 不 segfault。
4. **obs_mode='pixels' 是正确性 blocker**：同场景不同目标 suite（libero_goal）状态无任务 id → 矛盾监督。用 **vlm 模式** 或 **--task_id_onehot**。（注：vlm 的 `rep[:,-1,:]` 是 PAD 位近似均值池化，文本 token 被数百图像 token 稀释，libero_goal 上任务条件化可能偏弱；libero_10 不同场景则靠图像天然区分——所以 8 条件建议**统一 --task_id_onehot=N**隔离消融轴。）

## 算力：多任务把数据摊到 N 任务，需更多步（5 任务 ~40-50k / 全 10 任务 ~60-80k）；eval 成本 N× → 用 tasks_per_eval / eval_episodes 控界。本地 8 条件**串行**；大服务器用 run_ablation8_parallel.sh。`--checkpoint_interval 50000`。

**工程量**：代码 0.5-1 人日 + 验证/标定 1-2 人日 ≈ 2-3 天到可出图；正式全量训练多日（建议移多卡服务器或先 5 任务子集）。

---

# 方向二：换到「长程 + 精细操作」的新 benchmark

## 门控铁律（先读）：**没有任何现成 pi0 checkpoint 同时满足「新 + 长程 + 精细 + 有 headroom」**

- libero_10 长程但对 pi0_libero **in-distribution**（无空间）；
- ALOHA 转方块精细但**已是存量任务**、非长程；
- 直接拿转方块 ckpt 跑别的任务 → OOD + prompt 错 → base≈0%，**DSRL 稀疏奖励无成功可 bootstrap，学不动**。

→ **任何真正更难的新基准都要自训一个 pi0（SFT）。而本机 24 GB 只能勉强 LoRA、不能全量。** 这是整个方向二的核心成本。

## 四个候选（按"能不能拿到 checkpoint"分层）

| 选项 | 轴 | sim | pi0 checkpoint | 集成成本 | 净评价 |
|---|---|---|---|---|---|
| **A. ALOHA Bimanual Insertion** | 精细/接触 | MuJoCo（已有） | ❌ 需自 SFT（`lerobot/aloha_sim_insertion_human`，克隆 pi0_aloha_sim 配方换一行 repo_id） | **极低**（6 处 elif，reset/step/reward 靠 'aloha' 子串零改） | **本机自 SFT 最省的一条**；但非极端长程 |
| **B. LIBERO-Long 排除式再微调** | 长程 | MuJoCo（已有） | ❌ 需自 SFT（剔除 libero_10 重训 pi0，令其 OOD） | **零**（仅 `--task_suite libero_10`） | 覆盖纯长程；但数据按 suite 过滤繁琐、headroom 不确定（pi0 泛化强，可能只掉到 ~70%） |
| **C. RoboCasa** | 长程+精细+语言 | robosuite/MuJoCo（复用 LIBERO 栈） | ⚠ 社区有 pi0/pi0.5 baseline，但**本 loader 未验证**、数据需转 LeRobot、可能要写 RoboCasaInputs | 高（按周计） | 唯一原生同压两轴的**真新**基准；远期拉伸 |
| **D. 社区 RLinf/πRL checkpoint** | 视 ckpt | CALVIN(PyBullet)/ManiSkill3(SAPIEN)/LIBERO | ⚠ HF 上有现成 pi0 SFT（如 `RLinf-Pi0-CALVIN-ABC-D-SFT`），但**格式/norm_stats 未在本 fork 验证**，且 CALVIN/ManiSkill 是外来 sim | 中-高 | 唯一"可能免自 SFT"的路；但加载与外来 sim 双重不确定 |

## 主推：**A. ALOHA Insertion**（精细轴，本机自 SFT 最省）+ 互补 **B. LIBERO-Long-excl**（长程轴）

理由：insertion 环境已内置 `gym_aloha`（`InsertionTask`，分阶段奖励 0..4，4 个顺序子目标 touch→lift→peg-socket 接触→pin 插入），数据集已是 LeRobot 格式，**训练与 DSRL 推理复用同一套 `LeRobotAlohaDataConfig`+`AlohaInputs/Outputs` → 输入契约天然自洽**（规避了"obs_to_pi_zero_input 与冻结 ckpt 契约对齐"这一最大风险）。集成只需：

- `train_sim.py`：3 处 elif（DummyEnv state_dim=14；env 构造 register `AlohaInsertion-v0` task='insertion' + env_max_reward=4 + max_timesteps=400；pi0 dispatch 指向新 config）。
- `train_utils_sim.py`：3 处把 `== 'aloha_cube'` 放宽为 `in ('aloha_cube','aloha_insertion')`（obs_to_img / obs_to_pi_zero_input / obs_to_qpos）；reset/step/reward/直方图**零改**。
- `openpi/src/openpi/training/config.py`（**两处 checkout 都要**）：克隆 `pi0_aloha_sim` → `pi0_aloha_sim_insertion`（repo_id 换 insertion，default_prompt 待核对）+ LoRA 变体 `_low_mem`（仿 `pi0_libero_low_mem_finetune` 的 gemma_2b_lora + freeze_filter + ema_decay=None）。
- 脚本：复制 `run_aloha.sh`→`run_aloha_insertion.sh`、`run_ablation8_aloha.sh`→`run_ablation8_aloha_insertion.sh`（仅改 --env）。

## ⚠ 修正后的强制步骤 / 风险（红队核实）

1. **【blocker】checkpoint 不存在，必须自 SFT，且本机算力未验证**：full FT 70 GB 本机不可行；LoRA 22.5 GB 在 24 GB 上勉强、当前已占 5 GB → **可能塞不下**。→ **Stage-0 go/no-go**：先用小 debug config 确认 openpi **训练**（不只推理）能在本机端到端跑；确认 LoRA 在无 GPU 争用下能装下（`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`）；否则**租 ≥80 GB 卡做全量 FT**，把冻结 ckpt 下载回本机（DSRL 前向仅 ~8 GB，轻松）。
2. **【major，综合器漏了】必须跑 `compute_norm_stats`**：`create_trained_policy` 从 `checkpoint_dir/assets` 读 norm_stats（policy_config.py:64）；**转方块的 norm_stats 不能复用于 insertion**（state/action 分布不同 → 冻结 pi0 收到错归一化输入 → base 静默塌到 ~0）。→ 顺序：加 config → 下数据 → `openpi/scripts/compute_norm_stats.py --config pi0_aloha_sim_insertion` → SFT → **断言输出 ckpt 含 `assets/.../aloha_sim_insertion_human/norm_stats.json`** 再指给 DSRL。
3. **【major】base 落点未知**：insertion 比转方块难，20k 步 + ~50 示教 + LoRA 可能欠拟合。若 base<~10%，DSRL 无成功可 bootstrap → 学不动。→ **DSRL 8-condition 前硬门控**：`i==0` Gaussian base eval 必须落在 **~20-70%**，否则先补 SFT / 全量 FT，别投多日消融。
4. **【major，修正综合器事实错误】**：综合器称"检验排序是否 vs 转方块翻转"——**错**。8-condition 消融跑的是 **LIBERO-90 task 46**，**转方块从未做过消融**。→ 正确表述：insertion 8-condition 对照的是 **LIBERO-90 #46 的排序**（vlm/na 单项最佳 0.98、三叠最差 0.84、dual buffer 有害）。
5. **数据未在本地**：`lerobot/aloha_sim_insertion_human` **不在 ~/.cache**（综合器误称已有），需从 HF 下载（走 clash 代理，可能限速）；下后**核对 schema**：`observation.images.top / observation.state[14] / action[14]` 与 `LeRobotAlohaDataConfig` repack 键一致（config.py:212-224）。
6. **default_prompt 必须与数据集 task 标签逐字一致**（SFT 与推理都靠它），`'Insertion'` 只是占位待核。

## 💡 最省的第一步（红队亮点，强烈建议）：先跑 **transfer_cube（转方块）8-condition** 作桥

- **checkpoint 已在本地**（pi0_aloha_sim，~12 GB），`--env aloha_cube` **已接线**，**零集成、零 SFT**。
- 转方块是比 libero **更精细（14-DoF 双臂）** 的任务，且**从未被 8-condition 消融过** → 免费拿到"第 2 个更难数据点"，验证 §0 的"方法差异在样本效率"假设。
- 本机可直接串行跑（`run_ablation8_aloha.sh`）。**这是投入任何 SFT 之前性价比最高的动作。**

## 算力：一次性 SFT 是唯一吃满显存的步骤（LoRA-3090 ~4-8 GPU·时 / 租 A100 全量 ~3-6 GPU·时），**勿与 DSRL 训练同时跑**；8 条件因 31 GB RAM **必须串行**（每进程常驻一份 3.3B pi0，勿用 *_parallel.sh）。

**工程量**：insertion 环境集成 0.5 天 + config 克隆 0.5 天 + SFT 1-1.5 天（含下数据/算 norm_stats/LoRA 或租卡/base 验证）+ 8 条件串行 3-5 天纯算力 ≈ **1 周工程 + 数日算力**。B(LIBERO-Long-excl) 追加 2-3 天。C(RoboCasa) 按数周。

---

# 推荐排序（把风险从低到高排）

1. **【本周即可，零风险】方向一 pilot + transfer_cube 桥**：
   - (a) 跑 libero_10 per-task base **probe**（Stage-0 门）；跑 **2-3 任务多任务 pilot** 看 8 方法在样本效率/hard-subset 均值上能否分开；
   - (b) 并行地把 **transfer_cube 8-condition** 串行跑起来（免费的第 2 个更难数据点）。
   - 两者都不需要新 checkpoint，直接回答"更难/多任务到底能不能区分方法"。
2. **【若 pilot 证明可分】方向一全量 libero_10 多任务** + §0 的 AUC/样本效率指标 + hard-subset headline。
3. **【要真·新基准且愿意投 SFT 算力】方向二-A ALOHA Insertion**（精细轴，本机自 SFT 最省，先过 Stage-0 算力门 + base 门）；如需长程再加 **B LIBERO-Long-excl**。
4. **【远期/高投入】C RoboCasa**（两轴全覆盖的真新基准，先落实并验证一个 pi0 checkpoint）或 **D 试社区 RLinf ckpt**（可能免 SFT，但加载 + 外来 sim 双不确定）。

> 一句话：**先用"多任务 + transfer_cube + 换样本效率指标"这三件零成本的事去区分方法；只有当你确实需要一个全新的长程/精细基准、且能接受一次 pi0 自 SFT（本机 LoRA 或租卡）时，才走 ALOHA Insertion / RoboCasa。**

---

## 附：8-condition 矩阵在新 env 上零摩擦复用

`run_ablation8_aloha.sh` 除 `--env` 与 ALOHA 超参外完全 env-无关，8 条件（baseline / vlm / buf1 / buf2 / na / vlm_buf2 / na_buf2 / vlm_na_buf2）`variant_flags()` 原样迁移；`env_act_dim=14` 自动，NA 的 `extra_fields=(query_freq,14)` 自洽。多任务版同理，8 条件不动，只在 COMMON 加多任务 flag。
