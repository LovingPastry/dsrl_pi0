# Robotics Idea Discovery Report — DSRL-pi0 → ICLR/CoRL

**Direction**: 改进 DSRL-pi0（在冻结的 pi0 flow-matching VLA 的 latent-noise 空间做 SAC 强化学习），目标顶会 ICLR 或 CoRL。
**Date**: 2026-07-06
**Pipeline**: ultracode 31-agent 工作流（survey + gap + ideate + 对抗双审）→ area-chair 重建 Top-3 → novelty-check → Codex(gpt-5.5) 外部评审
**约束**: pi0 全程冻结/黑盒；单卡 RTX 3090 24GB + 31GB RAM（不可微调 pi0，仅 pi0_libero / pi0_aloha_sim）；sim-only（LIBERO + ALOHA-sim），无真机；headline 指标 = 样本/查询效率、AUC、success-at-budget；不改 diffusion/flow 架构。

---

## Robotics Problem Frame
- **Embodiment**: 单臂（LIBERO, Franka）+ 双臂（ALOHA-sim, 14-DoF），sim only
- **Task family**: 语言条件桌面操作（pick-place、长程多阶段、精细接触）
- **Obs / Action**: RGB(多视角) + proprioception + language ；动作 = SAC 在 32-d latent-noise 上决策 → 冻结 pi0 flow-ODE 解码成 action chunk（query-level semi-MDP）
- **Learning regime**: 冻结生成式策略之上的 off-policy RL（SAC），黑盒 steering
- **Assets**: DSRL-pi0 代码库、LIBERO/ALOHA-sim、pi0_libero/pi0_aloha_sim 冻结 ckpt、DSRL 原论文（`docs/2506.15799v2.pdf`）、8-condition 消融历史
- **Compute**: 单卡 3090；每 env step = 一次昂贵的 50-step pi0 ODE 解码；SAC 头 ~0.5M 参数（梯度步几乎免费）
- **Contribution type**: method + diagnosis + evaluation protocol（非新架构、非真机 sim2real）

---

## Landscape（要点，完整见 ultracode 工作流输出）
调研覆盖：**VLA 权重 RL 微调**（iRe-VLA, VLA-RL, GRAPE, ConRFT, RLDG, RIPT-VLA）——共识是直接更新 VLA 权重不稳、贵、易灾难性遗忘，DSRL 靠"完全冻结 + 噪声空间 RL"规避；**冻结策略的 latent/action-space RL**（DSRL=arXiv:2506.15799, ZPRL, 残差 RL）；**offline→online 采样效率**（RLPD, Cal-QL, WSRL, 对称采样）；**免密集奖励的 reward/value**（VIP, LIV, GVL, Diffusion Reward）；**采样效率与可塑性**（REDQ, DroQ, CrossQ, Primacy-Bias/resets, SR-SPR, BBF, ReDo）；**评测严谨性**（rliable IQM/CI）。

**9 个带代码 hook 的 gap**（可改进面 × 前沿交叉）已产出；其中"表征/奖励塑形簇"先验过密，"采样效率/评测/探索簇"（gap 5/7/8）更冷门更硬。

---

## Ranked Ideas

### Idea #1 — RECOMMENDED（经 Codex 外审后重定位）
**English title (for venue)**: *Query-Efficient Reinforcement Learning for Frozen Generative Robot Policies — when action decoding is expensive and the trainable head is tiny, high-UTD off-policy RL is the compute-optimal regime, but only with correct update accounting, overestimation control, and plasticity maintenance.*

- **一句话**: 在 DSRL 里，每个 env step 触发一次昂贵的冻结 pi0 ODE 解码（主导 wall-clock），而 ~0.5M 参数 SAC 头的梯度步几乎免费 —— 这**反转**了 REDQ/DroQ 的"高 replay 很贵"前提。正确的优化目标应是**"每次昂贵 VLA query 的成功率"**，而非每梯度步。
- **Bottleneck**: 冻结 VLA 下 RL 的 compute-vs-sample 计算模型被反转，vanilla DSRL 三处没利用：(1) 10-critic 用 **mean** 归约（无 subset-min 控高估、无 optimism/disagreement 促探索）；(2) **真 bug**：`num_gradsteps = len(traj['rewards']) * multi_grad_step`（`examples/train_utils_sim.py:334-337`）把更新数绑到轨迹长度 → 成功的短轨迹更新更少，UTD 与数据质量**反相关**；(3) 小头在稀疏非平稳目标上高 UTD 训练 = 可塑性丧失高危区，冻结 3.3B backbone 是护还是饿其可塑性未知。
- **Method**: (a) UTD 与轨迹长度解耦，按 env-step 定固定 G，扫 G∈{1,4,8,16,32}；(b) 内存友好稳定器 DroQ / CrossQ；(c) ensemble 归约 mean→REDQ 随机 subset-min；(d) 可塑性维护 SR-SPR soft reset + ReDo，跟踪 dormant fraction / effective rank；(e) x 轴用 **env-step + wall-clock**，把反转-算力论点量化。
- **Falsifiable**: P1 解耦 UTD 在**等总梯度步**下仍提升 AUC；P2 存在交叉点——mean 高 UTD 崩、DroQ/CrossQ+subset-min+reset 随 G 持续 scaling；P3 dormant fraction 随 UTD 升、被 reset/ReDo 压制且与 AUC gap 相关。
- **Pilot**: sim（LIBERO-90 hard-headroom + ALOHA-sim），单卡 3090 可跑，pi0 冻结。
- **Novelty（Codex 认可的那句）**: 冻结生成式策略 RL 的正确目标是 success-per-VLA-query，不是 success-per-gradient/parameter —— 需用**成本曲线图**（pi0 ODE 解码时间 vs 更新时间 vs GPU 利用）坐实。
- **Venue**: **CoRL 主投**（borderline→accept-lean 可达），ICLR 需额外抽出"昂贵黑盒解码器下高 UTD 的一般性 scaling law"才够。
- **Reviewer score（Codex）**: ICLR 5/10 reject-lean；CoRL 6.5/10 borderline。

### Idea #2 — 捆绑进 #1（评测协议 + 鲁棒性恢复）
现在代码测的是"最终成功率 vs **梯度步** + **随机**探索策略 + **轮转**多任务子集 + 100% early-exit"——正好掩盖自己的样本效率差异。补齐：deterministic `eval_actions`、success vs **env-step**、AUC / env-steps-to-{50,80}% / success-at-budget、rliable IQM + bootstrap CI、固定 hard-subset headline，外加 **LIBERO-PRO/Plus 扰动下的鲁棒性恢复**测试（噪声空间 steering 能否恢复被扰动打崩的冻结 pi0）。Venue: ICLR 方法学/CoRL 协议。**Codex 明确建议 #1 必须和 #2 捆绑，否则 #1 单独像"optimizer patch / ablation-heavy"**。

### Idea #3 — 暂缓（噪声盒定向探索 UCB/RND）
暴露 Q_std 做 UCB bonus、或在已缓存 pi0 prefix 上加 RND。**Codex 建议先不要加**：在修好 UTD/稳定性前引入会 scope-creep 且与主线混淆；仅当修完后仍明显缺探索再补。

---

## Eliminated Ideas（对抗双审否掉）
- **奖励塑形 / critic-input 五连**（potential-based progress shaping from pi0 features、rank-1 noise seed、consequence critic 等）——先验太密（VIP/LIV/SORS/PROGRESSOR/Rank2Reward/Diffusion Reward）；**Ng-1999 只保证最优不变、不保证样本效率**，且在线训练的 Φ 需动态 PBRS；consequence-critic 与 DSRL-NA 的 Q_A 信息冗余。全部 survives=False（3–4/10）。

---

## Evidence Package for the Top Idea（Codex 的最小可接受包）
- **Baselines**: 冻结 pi0 zero-shot；vanilla DSRL(mean+长度耦合 UTD)；bug-fixed DSRL(固定 UTD+mean)；REDQ-DSRL(固定 UTD+subset-min)；DroQ- 或 CrossQ-DSRL（择一，除非算力够）；full(固定 UTD+subset-min+最佳稳定器+reset/ReDo)；DSRL-NA(若已稳定)。
- **Ablations**: UTD 扫 {1,4,8,16,32}；长度耦合 vs 固定 UTD 在**等 env-step**；**等总梯度步**对照（隔离 bug）；mean vs subset-min（同 ensemble）；DroQ/CrossQ on/off；reset/ReDo on/off；每配置 **wall-clock 记账**；可塑性诊断（dormant fraction、effective rank、Q-bias 代理、TD-error 分布）。
- **Tasks**: ≥8 个 base pi0 成功率 30–80% 的 LIBERO 任务（饱和 >90% 移出 headline、单独报非回归）；2 个 ALOHA-sim（若稳定跑）；**任务子集在看结果前预注册**。
- **Seeds**: ≥5（headline 方差大则 10）；deterministic eval，≥20 episodes/checkpoint。
- **Make-or-break 图**: **AUC/IQM vs UTD G，叠加 wall-clock-to-threshold**，对比 vanilla-mean / fixed-UTD-mean / REDQ-subset-min / full，在 hard LIBERO subset 上。若高 UTD 不提升 AUC 且不省 wall-clock，则论点被证伪。
- **必须补的缺口**: 直接证明"pi0 ODE 解码在所有 G 下主导成本"；bug 的等梯度对照；Q 高估诊断；对 action/noise box 尺寸与 SAC 温度的敏感性；高 UTD 下 replay 数据分布分析；semi-MDP chunk 折扣的清晰处理。

## 需预先反驳的致命点（Codex）
- "饱和任务上增益消失"→ hard-headroom 作 headline，饱和任务仅报非回归。
- "G=32 时 wall-clock 前提不成立"→ 实测；若 G=32 wall-clock 反亏，把卖点从"wall-clock-efficient"改为"**query-efficient**"。
- "这就是 REDQ on DSRL"→ 必须给成本模型图 + DSRL 特有失败分析（长度耦合欠训、Q 归约、semi-MDP chunk、冻结解码器延迟）。
- "可塑性论点空泛"→ 需干预证据（高 UTD 升 dormancy、reset 降之，控制更新数后预测 AUC）。
- "bug fix 解释了一切"→ 用等梯度/等 env-step 双对照分离 bug 与稳定器。
- "梯度步不免费因 pi0 特征重算"→ 合法处缓存冻结特征，报 with/without。
- "CrossQ/BatchNorm 在非 iid replay / 小 batch 下失效"→ 加 batch-size 敏感性与 train/eval mode 处理。
- "sparse reward 下 subset-min 过悲观"→ 跟踪欠估与探索失败，比较 M=2 vs mean。

## ⚠ 待人工核实的 prior art
Codex 提到 **ForesightFlow / SteerGenPO** 两篇"VLA/flow 策略改进"工作——**尚未独立验证是否真实存在**，投稿前须查证（Codex 偶有虚构 plausible 标题）。其余引用（DSRL 2506.15799、REDQ 2101.05982、DroQ 2110.02034、CrossQ、Primacy-Bias 2205.07802、ReDo 2302.12902、rliable 2108.13264、pi0 2410.24164、LIBERO 2306.03310）均为真实。

---

## Next Steps
- [ ] 核实 ForesightFlow / SteerGenPO 是否存在；补一轮针对"frozen generative policy RL + 采样效率"的精确查新。
- [ ] Stage-0 成本模型 pilot：实测 pi0 ODE 解码时间 vs SAC 更新时间 vs GPU 利用，坐实"query 主导成本"（这是全论文的地基，先做）。
- [ ] 实现 UTD 解耦 + 等梯度对照，先跑单任务验证 P1（bug 效应）。
- [ ] 若 P1 成立 → 铺开 baselines/ablations（先 5 任务 pilot 再全量），产出 make-or-break 图。
- [ ] 用 `/auto-review-loop` 对"Query-Efficient RL for Frozen Generative Robot Policies (#1+#2)"迭代打磨。

## 中文总结
最终推荐 = **把 #1 与 #2 捆绑成一篇"面向冻结生成式机器人策略的 query-efficient RL"**。卖点不是"套用 REDQ/DroQ/ReDo"，而是揭示并量化一个被反转的优化目标：**冻结 3.3B pi0 的每次 ODE 解码极贵、而 0.5M SAC 头的梯度步近乎免费，所以正确度量是"每次 VLA query 的成功率"**；并顺带修一个真 bug（更新数被轨迹长度耦合，导致成功短轨迹欠训）。定位 **CoRL 主投（6.5/10 borderline，可推到 accept-lean）**，ICLR 需再抽出一般性 scaling law（当前 5/10）。全程 pi0 冻结、单卡 3090 可跑。地基性的第一步是**成本模型 pilot**——若"pi0 query 主导成本 + 高 UTD 提升 AUC"不成立，整个论点被证伪。
