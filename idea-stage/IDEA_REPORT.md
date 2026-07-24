# Robotics Idea Discovery Report — Human-Intervention Data Valuation in Noise-Space VLA RL

**Direction**: 在 DSRL 框架（冻结 pi0 flow-matching VLA + 32 维 latent-noise 空间的 SAC）下，当人类通过 **teleop 整段接管** 介入真机 RL 时，应如何对"策略产生的数据"与"人类接管产生的数据"做 **价值/信用分配（credit assignment）**：统一处理，还是给人类帧更高价值？
**Updated conditions**: 有云服务器（可全量微调 pi0）+ 大量真机可验证。CoRL 已截稿；可投 ICRA / ICLR / CVPR（下半年）或 RSS / IROS（明年初）。
**Date**: 2026-07-07
**Pipeline**: 24-agent ultracode workflow — 6-cluster HIL survey (web-verified) → taxonomy synthesis → 4-angle ideation → 3-lens adversarial vetting (novelty / feasibility / real-robot-evidence) → area-chair memo.
**Note**: 取代前一版 query-efficiency 方向（存于 `IDEA_REPORT_v1_query_efficiency.md`），因用户条件变化（真机 + human-in-loop）而重定向。

---

## 0. 直接回答用户的原始问题

> **问：** 人类接管数据该和策略数据"统一处理(uniform)"，还是"给人类帧更高的 Q(fixed upweight)"？
>
> **答：两个极端都错。正确做法是"差异化，但要解耦 + 自适应"。**

- **纯 uniform（DSRL-NA 现状）被支配**：它只靠稀疏奖励慢慢回传信用，把"此处策略差 / 人类成功纠正"这个**可靠信号完全没定价**。
- **固定更高权重被证否**：50/50 双 buffer、IWR/Sirius 固定类权重、UniSteer 固定 noise-BC pin——正是本代码库发现**会损害后期微调**的做法；IntervenGen 独立显示固定加权比数据增广差 **48–74%**。在噪声空间**更糟**：被加权的接管帧，其噪声表示 ẑ_h 是通过**有损逆变换伪造**出来的。
- **真正答案**：**只在接管数据可靠的那个轴上给高权重，在被伪造的轴上弃权**。
  - 可靠轴 = 动作空间（"策略此处差" + "人类纠正 a_h 好"）→ 满强度路由进 DSRL-NA 的动作原生 critic **Q_A（零求逆**，已核实 `update_qa_critic` 直接吃 `env_actions`）；并以**相对偏好（human ≥ policy）**而非绝对值断言。
  - 伪造轴 = 逆出来的噪声 ẑ_h、噪声 critic Q_W、噪声 BC → 按**每帧逆变换置信度 c(s)** 降权/弃权。
  - "高多少" = **每帧、随逆置信度缩放**的量，绝非固定比例。

---

## 1. Robotics Problem Frame

- **Embodiment**: 单臂（LIBERO / ALOHA-sim 仿真；Franka DROID 真机，openpi websocket 远程 pi0）。
- **Task family**: 接触密集操作（pick-place / insertion / 长程重排），稀疏 −1/0 奖励，query-level semi-MDP。
- **Human interface**: **teleop 整段接管**——人类在近失败段接管，产出**原始动作 chunk a_h**；人类同时负责场景复原 + 成功/失败标注；训练时人类不能离开。
- **Policy interface**: 冻结 pi0 黑盒；SAC 在 32 维噪声种子上行动，噪声经 flow ODE 解码为动作。DSRL-NA 变体有动作原生 critic Q_A（蒸馏到噪声 critic Q_W）。
- **核心难点（DSRL 独有）**: 策略在**噪声空间**优化，人类在**动作空间**接管 → 接管帧**没有原生噪声标签**，必须 action→noise 逆变换（有损、一对多、确定性但不确定）。

---

## 2. 综述：人类接管数据的估值策略（web-verified, 13 类 / 26 方法）

全场收敛到一条 **UNIFORM ↔ DIFFERENTIAL** 光谱：

| 策略 | 代表方法 | value/credit 处理 | 噪声空间做过? |
|---|---|---|---|
| uniform 单 buffer | RLPD, **DSRL/DSRL-NA**, HIL-SERL(RL侧) | 统一 TD，信用只靠稀疏奖励回传 | 是（DSRL，仅自主数据，无人类通路）|
| 固定 50/50 对称采样 | RLPD, SERL, **HIL-SERL**, ConRFT | 结构性上采样，价值仍统一 | 是（结构上；固定比例正是 backfire）|
| 显式加权 BC | **IWR**, **Sirius**, DDPGfD | 固定类权重上采样人类帧 | UniSteer=按构造上采样 |
| intervention-as-penalty | **RLIF**, EGPO, HACO | 只惩罚被覆盖的策略动作，**只需时机** | **否**（最可迁移，从未在噪声空间试过）|
| proxy-value 硬 pin | **PVP**, HACO | 绝对 +1/−1 pin 人类动作 | 否 |
| advantage-weighted | AWAC, IQL, IBRL, AW-Opt | exp(A/λ) 加权，信任 critic 判断 | 否 |
| conservative/large-margin | CQL, **Cal-QL**, DQfD | 抬 in-support / demo 动作 Q | 否 |
| preference-relative | **PPL**, Sirius | 只断言 human ≥ policy，无绝对值 | 否（概念上最有前景）|
| coverage-instead-of-reweight | **IntervenGen**, FRS | 增广后统一训练（显示固定加权差 48–74%）| FRS=是 |
| gating-only | HG-DAgger, ThriftyDAgger | 价值信号花在"何时介入"而非"数据值多少" | 否 |

**DSRL 空白（精确）**: 上述**每一种差异化策略**都假设人类动作与 agent 优化在**同一空间**、标签**完全可靠**。DSRL 里 agent 优化噪声种子，接管产出的动作**无原生噪声标签**——这制造了一个**动作空间没有类比的置信不对称**：接管帧**动作可靠**（此处坏 / 人类纠正好）但**噪声不可靠**（ẑ_h 是伪造的、带误差的重建）。未被研究的是：**逆变换不确定性感知的、动作 vs 噪声解耦的**信用分配。而最可迁移的原语（RLIF 的 intervention-as-penalty，只给"此处策略差"的时机定价）**恰恰从未在噪声空间试过**——因为它是唯一绕开不可靠逆变换的方案。

**UniSteer（唯一噪声空间尝试）留下什么**: 它解决了**表示问题**（如何给接管拿到噪声标签 ẑ_h），但留下**估值问题**——(1) human≥policy 偏好只**隐式**通过模仿，无显式 advantage/penalty/偏好；(2) **每个 ẑ_h 都当等可靠 ground-truth**，无逆误差/分歧/一对多加权；(3) **不解耦**可靠动作信号与不可靠噪声标签；(4) 每步监督 pinning 是**固定结构上采样**——正是 backfire 那一家。FRS(2026) 同样"每个逆出的噪声等可靠、无不确定性加权"。

---

## 3. RECOMMENDED IDEA — CAPRI

**CAPRI: Confidence-Asymmetric Preference Credit for Teleop Takeovers in Noise-Space RL over Frozen Flow Policies**
（4 个候选中**唯一三个对抗视角全部存活**、meanScore 最高 **5.67**、新颖内核最锐。）

### 核心新颖内核（confidence asymmetry）
一个噪声空间接管，其 human≥policy 偏好的**两半以不同可靠度承载**：
- **被覆盖的策略噪声 ẑ_π 是原生已知的**（正是人类否决的那步）→ 完全可靠；
- **人类纠正是原始动作**，其 ẑ_h 只能靠有损 backward-ODE 逆变换重建 → 伪造。

CAPRI **分别给两半定价**，且是两个已知方案之间**可证的插值**：c→0 = RLIF-噪声版，c→1 = PVP/PPL-噪声版。

### 机制（一个按可靠侧拆分的非对称排序损失，作用在 Q 上）
1. **负半边（满强度、无伪造）**：接管态 s 把 `Q_W(s, ẑ_π)` **压低** = RLIF intervention-as-penalty 首次搬进噪声空间（ẑ_π 是策略自己发出的噪声，无需求逆）。
2. **可靠价值信用（满强度、无求逆）**：人类 chunk a_h 直接训动作原生 critic **Q_A**（吃存储的 `env_actions`），经已有 index-matched **Q_A→Q_W 蒸馏**传到噪声策略。（核实：`pixel_sac_na_learner.py` `update_qa_critic` L59 吃 `batch['env_actions']` L82，无求逆。）
3. **正半边（置信门控）**：K 步 backward-ODE 逆出 ẑ_h，只断言**相对序** `Q_W(s, ẑ_h) ≥ Q_W(s, ẑ_π) + m·c(s)`，**永不 pin 绝对 +1**。actor 不动（仍最大化 Q_W），所有估值只经信用分配进入，**无固定 buffer 比例、无固定 BC pin**。
4. **鲁棒性下界（吸收自 First-Do-No-Harm）**：正目标封顶在 base-pi0 可达值，使次优人类永不把学习拖到 base 以下——作为**经验安全指标**上报，**非形式化证书**（该证书claim已被杀）。

### 置信信号 c(s) —— 按代码核实重新定义（关键修正）
pi0 的 `sample_actions` 是**确定性 Euler ODE**（`dt=-1/num_steps`, `pi0.py:279`），**没有随机 solver**，且**只有一个冻结 checkpoint** → "stochastic-solver / flow-ensemble 分歧"**不存在**（这一点直接否掉 CRISP 与 Trust-the-Action 的置信设计）。c(s) 必须由：(i) round-trip 重解码残差 ‖decode(ẑ_h)−a_h‖，(ii) **噪声先验典型度**（离流形人类动作逆出高范数、低 N(0,I) 密度噪声——**主信号**），(iii) K 步数分歧（K vs 2K 步逆变换漂移），(iv) 一对多多重性。**先在仿真 held-out probe 上用真实重建误差验证 c(s) 再信任它。**

### 代码落点（均已核实）
在 `jaxrl2/agents/pixel_sac_na`（及 base `pixel_sac` critic_updater 的 negative-only 消融）加 hinge-margin；在 `update_qw_distill`（L96，蒸馏 (s,w,a) 别名对 L104）里用 c(s) 加权接管对；扩展 `env_actions` extra_fields（`train_sim.py:207`）加接管 flag、ẑ_π（actor 存的噪声）、c(s)；把 backward-ODE 逆变换实现为确定性 `sample_actions` 的逆（**目前缺失**）；baseline = 固定 50/50 `_mixed_iterator`（`train_utils_sim.py:250`，已记录的 backfire）。

### 可证伪假设
- **H1（系统）**：固定接管预算下，CAPRI 击败 {uniform DSRL-NA, UniSteer 固定 BC, 固定 50/50, RLIF-negative-only}，并**抹掉** 50/50 的后期回退。
- **H2（机制，载荷）**：随着**受控注入的逆误差**上升，CAPRI 对 c=1 消融的优势**单调增大**；若 c 无信息（CAPRI≈c=1≈RLIF）或固定 baseline 在同预算下追平，则**证伪**。

### Headline 指标
固定接管预算下的**成功率 AUC / return-AUC**（方法在天花板处收敛，分离在样本效率里——本库自身结论），**配对** H2 斜率 `dΔsuccess(CAPRI−c=1)/d(逆残差)` 必须**严格为正**。

### 场馆
**ICLR 为主**（算法向：在学出的噪声空间里做不确定性门控的相对偏好信用分配，可证插值 RLIF↔PVP；载荷 H2 只能在仿真测）。**RSS/CoRL 为辅**，真机演示收窄为"接管次数方向性下降、无安全回退"，**不承诺** n≤3 真机能支撑的统计 Pareto 优势。

---

## 4. Ranked Ideas（全部存活，均分 4.83–5.67）

1. **[5.67] CAPRI（value-credit 轴）— RECOMMENDED**。唯一无 lens 击杀；新颖内核最锐（confidence asymmetry + 可证 RLIF↔PVP 插值）；彻底去掉固定比例，直答本库 50/50-backfire；最佳机制隔离实验（H2）；Q_A 入口 + Q_W 蒸馏阀已存在于代码。
2. **[5.33] First, Do No Harm（do-no-harm 轴）**。头条"可证零回退证书"被 do-no-harm lens 击杀（初始即被证伪：DSRL 噪声 actor 随机初始化本就低于 base pi0；蒸馏滞后；Cal-QL 只在期望意义成立；硬件无法测）。其 `max(base,human)` 下界 + 安全监控指标**作为 CAPRI 组件存活**。
3. **[5.17] Trust the Action, Doubt the Noise（inversion-uncertainty 轴）**。与 CAPRI 同骨架，但头条押"逆变换不必要（ω=0 纯 Q_A 胜 UniSteer）"——代码确认的**搁浅风险**：`update_qw_distill` 需有效 (s,w,a)，ω=0 人类转移无 w 进不了 Q_W。**降为 CAPRI 的 ω=0 消融臂**。
4. **[4.83] CRISP（buffer-sampling 轴）**。新颖性最薄（4.5），可清晰拆成 UniSteer + 已发表的 uncertainty-weighted replay(UWAC/UDWER) + coverage annealing + 现有 Q_A。**coverage-annealing 作为 CAPRI 可选加件**。

**The defensible bundle**: CAPRI 吸收另 3 者精华为**自己的消融矩阵**（Trust 的 ω=0 纯 Q_A 臂 + u 验证协议；Do-No-Harm 的 max(base,human) 下界 + 安全指标；CRISP 的 coverage-annealing 臂），把三个对手变成自身消融。

---

## 5. Killed

- **用户问题的两个极端**：固定更高权重（50/50 / IWR / Sirius / UniSteer）被本库 backfire + IntervenGen 48–74% 证否；纯 uniform（DSRL-NA）被支配（可靠信号未定价）。
- **形式化"零回退证书"**（Do-No-Harm 头条）：初始即被证否 + 蒸馏滞后 + Cal-QL 只期望意义 + 硬件不可测 → 只存活为经验安全指标。
- **"逆变换不必要 ω=0"**（Trust 头条）：Q_W 蒸馏需有效 w → 信用搁浅 → 降为消融臂。
- **任何依赖"solver / ensemble 随机性"的置信信号**：pi0 是确定性 Euler ODE + 单 checkpoint → 不成立。
- **真机作为统计头条**（四者原始 RSS-primary 定位）：机制 H2 只能在仿真受控注入逆误差测；真机 4–6 臂 × 多种子 + 拴人不可行 → 真机降为方向性确认演示。

---

## 6. Evidence Package

### 仿真（头条，承载全部消融 + 载荷机制；冻结 pi0 只训小 SAC 头，24GB-3090 单卡可跑）
- **Env**: LIBERO + pi0_libero，选**真有 headroom** 的难-但-可学 OOD 任务（难度图里 base pi0 ~30–40% 的 libero_90 子集）+ ALOHA-sim + pi0_aloha_sim 作第二本体（仅有的两个可用冻结 checkpoint）。
- **模拟 teleop**: 低-ensemble-Q / 近失败探测器触发**脚本 oracle / 更强参考专家**；记录 a_h **与** 被覆盖的 ẑ_π。**关键修正**（回应 do-no-harm 的保真-循环质疑）：**不要**用干净 on-manifold oracle——把专家扰到 pi0 流形外 / 用**不同于 pi0** 的策略当"teleop"，让 c(s) 有真实方差，−c 门控才能分离。
- **Build**: backward-ODE 逆变换（确定性 `sample_actions` 的逆，目前缺失）；c(s) 由 round-trip 残差 + 噪声先验典型度 + K 步数分歧 + 一对多多重性构造；先在 held-out probe 对真值重建误差验证。
- **Arms**: B1 uniform 单 buffer(=DSRL-NA)；B2 固定 50/50(=HIL-SERL/RLPD backfire)；B3 UniSteer 固定 noise-BC（must-beat）；B4 RLIF-negative-only(=CAPRI c→0)；CAPRI-full；CAPRI-minus-gate(c=1)。
- **Ablations**: −c(c=1)、−negative-half、ω=0 纯 Q_A、Q_A/Q_W-split vs Q_W-only、coverage-annealing on/off、max(base,human) floor on/off，以及 **matched-mean 固定调度**（同平均接管比例、无逐样本自适应——隔离"自适应"与"仅平均权重更低"的关键混淆控制）。
- **H2 受控扫描**: ≥4 级注入逆误差，测 Δsuccess(CAPRI−c=1) 斜率必须严格为正（带 CI），否则门控（全部新颖性）被证伪。
- **Protocol**: ≥5 种子（单种子 eval 噪声 ±0.1，<3 不可解释）；长训以暴露后期回退；**预注册** last-20% 指标 + 查询效率 AUC + CI。
- **Metrics**: 主=固定接管预算下成功率/return-AUC；载荷=H2 斜率；安全次要=oracle **不会**介入的状态上的成功率（证负半边不压制好的策略噪声）+ 相对 50/50 baseline 的 last-N eval 方差（学习中的 do-no-harm）。

### 真机（方向性确认演示，**明确不是**统计头条）
- **前置工程（净新增、当前缺失、真成本）**: (1) 把 `PixelSACNALearner` + `env_actions` 采集移植进 `train_real.py`（当前仅 `PixelSACLearner`，无 Q_A、无 env_actions）；(2) 建 teleop 接管入口——中途人类接管、逐步 human/policy 标签、按 `query_frequency` 边界对齐的人类动作 chunk（当前真机环仅末端 1/0 成功标注 + 手动复位）；(3) 在 openpi websocket 加**服务端 backward-ODE 逆变换端点**（远程 3.3B pi0 只暴露前向 `.infer(noise=...)`），并测每次接管延迟。
- **Setup**: Franka DROID 走现有 websocket pi0 路径；人类接管近失败段 + 复位 + 标注（不能离开）；1–2 个接触密集任务；接管预算封顶（几百帧）。
- **Arms**: 仅 CAPRI vs 最强 baseline(UniSteer)，2–3 种子，**不**跑全矩阵。
- **Metrics**: 主=takeovers-to-threshold（到达成功门槛的人类努力）+ 最终自主成功率；安全=每阶段近失败 / e-stop 次数 + 无 checkpoint 低于 base pi0（经验 do-no-harm + 预注册回退护栏）。
- **载荷真机证据**: 证明**真人离流形纠正动作确实产生宽而有信息的 c(s) 分布**（非近常数），且显著高于/异质于干净脚本 oracle——这是仿真 oracle 无法演练的 confidence-asymmetry 前提；缺它则"trust the action, doubt the noise"动机未被验证。

---

## 7. Next Steps

- [ ] **Foundational pilot（先做，成本最低）**: 实现 backward-ODE 逆变换 + 在 held-out sim probe 上量 c(s) 与真值重建误差的相关性。**若 c(s) 无信息，CAPRI 的全部新颖性被证伪**——先证这一步。
- [ ] `/novelty-check` 对 CAPRI 最终措辞再核一遍（重点 vs UniSteer / FRS / PVP / RLIF / PPL）。
- [ ] 仿真 6-arm + 消融跑通（先 LIBERO 单任务打通管线，再扩难任务 + ALOHA-sim）。
- [ ] 仅在仿真 H2 斜率为正后，再启动真机前置工程（移植 NA 到 train_real + 接管入口 + 逆变换端点）。

## ⚠ 待核验
- **CAPRI 是本工作流综合出的新命题**，非已发表方法——需在实现前用 `/novelty-check` 再确认无撞车。
- UniSteer(2026)、FRS(2026)、DSRL(2506.15799)、RLIF/PVP/HACO/EGPO/IWR/Sirius/PPL/IntervenGen/Cal-QL 等均由 agent web-verified；个别 arxiv id 实现前请再核。
  - **UniSteer = arXiv 2605.10821**，标题是 *"Unified Noise Steering for Efficient Human-Guided VLA Adaptation"*（Junjie Lu et al., 2026-05-11）——**"UniSteer" 只是文内方法名，不在标题里**，且与 arXiv 2605.30076（同名的 LLM activation-steering 论文）撞名，按 "UniSteer" 直接搜会搜到后者。
  - **FRS = arXiv 2606.13675**，*"Improving Robotic Generalist Policies via Flow Reversal Steering"*（Tang, Chen, Wagenmaker, Finn, Levine, 2026-06-11）。
- 代码引用（`update_qa_critic`/`update_qw_distill`/`sample_actions:279`/`train_sim.py:207`/`train_utils_sim.py:250`/`train_real.py`）由 agent 核过，动手前再对一遍行号。

---

## 完整产出
- 综述分类表 + 4 idea + 全部对抗 verdict + memo（JSON）：`scratchpad/hitl_full.json`
- 最终 memo（JSON）：`scratchpad/hitl_memo.json`
- 工作流原始输出：`tasks/wtk2pvvia.output`
- 工作流脚本：`workflows/scripts/human-intervention-data-survey-wf_fa9fd455-f6f.js`
