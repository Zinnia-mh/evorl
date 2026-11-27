# Fitness Admission：经验覆盖与时效性轻量诊断

## 范围与可复现性

- 只读数据源：`/disk1/zmh_projects/evorl_artifacts/fitness_experience_admission/seed4099-cp100-v1`
- 仓库工作树：`/disk1/zmh_projects/evorl_worktrees/fitness-experience-admission`
- 核对时分支与 HEAD：`codex/fitness-experience-admission@dd52c1fce45de214eec4ca05edbb581d48cea0ba`；本地与远程一致，开始分析前工作区干净。
- 冻结算法提交：`fb534b3b76f36a252025d90f54013482aa974da1`；seed4099 runner 提交：`05e06b31b4a1d82aea6e72a7095c0fbb0d034c2b`。
- 原实验 `comparison.json` 为 `COMPLETE`，`verification.json` 为 `PASS`。三个 arm 均从同一 CP10 状态 SHA-256 `49c79e4201aa38bf8b6a7d0beddec4840b5110a7bc72798a34f8fffa140e11b8` 分叉。
- 本阶段没有 rollout、优化器更新或训练。完整 CP100 checkpoint 按原 GPU backend 只读恢复；覆盖统计在 host NumPy 上完成并写入新的分析目录，原产物未修改。

本阶段查阅的实现文件：

- `evorl/replay_buffers/replay_buffer.py`
- `evorl/replay_buffers/source_aware_replay_buffer.py`
- `evorl/algorithms/erl/erl_td3/erl_ga.py`
- `evorl/algorithms/erl/erl_td3/erl_td3_workflow.py`
- `evorl/algorithms/erl/erl_td3/erl_utility_replay.py`
- `evorl/algorithms/erl/erl_td3/source_replay.py`
- `evorl/algorithms/erl/erl_td3/source_residency_batch.py`
- `evorl/algorithms/erl/erl_td3/utility_replay_training.py`
- `evorl/algorithms/erl/erl_td3/utility_replay_checkpoint.py`
- `scripts/fitness_experience_admission.py`
- `scripts/run_fitness_experience_admission.py`
- `scripts/run_fitness_admission_seed4099.py`
- `tests/test_fitness_experience_admission.py`
- `docs/fitness_experience_admission_cp100.md`
- `docs/fitness_experience_admission_seed4099.md`

## 已有性能事实

| Arm | AUC20–100 (M) | 相对 A | Return100 | 首次 Return ≥350 |
|---|---:|---:|---:|---|
| A | 777.590 | — | 366.61 | gen50 / 1.64M steps |
| F-Random16 | 619.282 | -20.36% | 363.34 | gen60 / 1.97M steps |
| F-Top16 | 579.997 | -25.41% | 366.73 | gen90 / 2.96M steps |

Top16 比 Random16 低 6.34%。Top16 的 admitted Fitness 均值为 101.90，rejected 为 -11.08，且每代都精确选择 Fitness Top16；Random16 两侧均值为 53.67/52.24。因此 seed4099 的负结果不是准入排序没有执行。

## Replay 组成与驻留年龄

CP100 的 sidecar 与 payload 使用同一环形索引。`source_generation` 是每条 EA transition 的权威生成代数，RL transition 使用独立 origin 标记；物理槽位顺序在环绕后不代表时间。

| Arm | EA / RL transition | EA / RL 比例 | EA 年龄均值/中位/最大 | 最近 5 代 | 之前 25 代 | ≥30 代 |
|---|---:|---:|---:|---:|---:|---:|
| A | 969k / 31k | 96.9% / 3.1% | 14.64 / 15 / 30 | 160k (16.51%) | 800k (82.56%) | 9k (0.93%) |
| F-Top16 | 941k / 59k | 94.1% / 5.9% | 28.91 / 29 / 58 | 80k (8.50%) | 400k (42.51%) | 461k (48.99%) |
| F-Random16 | 941k / 59k | 94.1% / 5.9% | 28.91 / 29 / 58 | 80k (8.50%) | 400k (42.51%) | 461k (48.99%) |

过滤组每代评估 32 个个体，但只写入 16,000 条 EA transition；RL 仍写入 1,000 条。A 的驻留直方图显示每个完整代有 32,000 条 EA transition。硬准入因此同时造成：EA 写入率减半、驻留跨度约翻倍、最近经验质量减半、RL 驻留比例上升。Top16 与 Random16 的年龄和 origin 组成完全相同，所以这些共同变化不能解释二者之间的 6.34% 差异，但可以解释二者共同落后 A 的候选机制。

## 等样本覆盖诊断

三个 CP100 Buffer 都保存了逐条 transition。分析对每组无放回固定抽取 100,000 条 EA transition，使用固定 seed `20261127`。真实下一状态取自 TD3 实际使用的 `extras.env_extras.ori_obs`，终止标记取自 `extras.env_extras.termination`。指标使用三组 pooled 标准化：协方差有效秩、50,000 个固定随机 transition 对的距离，以及固定 4D 高斯投影/pooled decile 网格占用。它们是可复现的描述性快照，不是独立训练样本或因果检验。

| 特征 | 指标 | A | Top16 | Random16 |
|---|---|---:|---:|---:|
| state | 有效秩 | 5.002 | 4.535 | 5.186 |
| state | 10k 网格占用 | 4,624 | 4,315 | 4,687 |
| action | 有效秩 | 1.977 | 1.811 | 1.980 |
| action | 平均成对距离 | 1.976 | 1.564 | 1.717 |
| transition `(s,a,r,s'-s,done)` | 有效秩 | 8.019 | 6.816 | 8.249 |
| transition | 10k 网格占用 | 6,071 | 5,810 | 6,020 |
| transition | 平均成对距离 | 6.265 | 5.529 | 5.482 |

Top16 的 state、action 和 joint-transition 有效秩均低于 A/Random16，说明 Fitness 硬筛选与更窄的已驻留行为/transition 覆盖相符。Random16 的有效秩和网格占用却总体接近 A，但 AUC 仍比 A 低 20.36%；所以这些简单覆盖指标不能单独解释共同减速。成对距离也受三个训练轨迹已经分叉和 pooled 标准化影响，不能解释为某一组“本来就更探索”。

## 证据等级

### 已证实

1. Top16 排序和写入掩码按设计执行，未准入 transition 不在 Buffer。
2. 硬半准入将 EA 驻留年龄约翻倍，并把最近 5 代 EA 质量从 16.51% 降到 8.50%，同时将 RL 驻留比例从 3.1% 提到 5.9%。
3. Top16 CP100 的简单覆盖指标低于 A/Random16；Random16 覆盖接近 A 但仍明显慢于 A。
4. 三组训练交互、Actor/Critic 更新、评估预算、初始 CP10 和最终训练/GA key 已审计一致。

### 存在混杂

- A 与两个过滤组的写入率、驻留年龄、EA/RL 驻留比例、具体 transition、后续策略轨迹和实际 uniform 抽样序列同时改变。二者都落后 A 与“旧经验比例上升”一致，但不能证明旧经验导致性能下降。
- Top16 与 Random16 控制了写入量和驻留年龄，却保留了所写内容的差异；Top16 的较低覆盖与额外负收益方向一致，但只有一个 seed 的 CP100 快照，不能作因果结论。
- 当前 checkpoint 没有反事实信息，无法知道被丢弃 transition 若写入后是否会改善 TD3。

### 待验证假设

1. TD3 对近 5 代 EA 经验的需求高于更旧的 EA 经验。
2. 维持全部 32 个 EA 的写入覆盖，同时在采样时提高近期质量，能避免硬准入的覆盖损失。
3. Top16 的额外退化可能与 episode Return 筛掉行为多样性有关；当前结果不足以区分这一机制与训练轨迹随机分叉。

结论：现有数据支持“时效性值得直接干预验证”的前提，但不支持“时效性已经被证明”。

## 5+25 时间分层采样的最小方案

### 可行性

现有 `SourceAwareReplayBufferState` 已逐 transition 保存 `origin_type`、`source_token` 和 `source_generation`，且 payload/sidecar 同索引环绕写入，无需改变物理 Buffer。现有 weighted Replay seam 已具备三个关键能力：

1. `draw_baseline_indices` 与原 `ReplayBuffer.sample` 使用同一 training replay key 产生基准索引；
2. `build_origin_slot_masks` 区分该批真实 EA/RL 槽位；
3. overlay 仅替换 EA 索引，RL 索引逐项保持不变，并使用独立、可 checkpoint 的 intervention key。

最小实现应新建独立实验 workflow，而不是修改冻结 Fitness Admission：

- 所有 32 个 EA rollout 继续写入同一 Buffer；RL 写入、GA 和 TD3 更新不变。
- 当前代数使用与 provenance 相同的 active generation（`iterations + 1`），年龄为 `active_generation - source_generation`。
- resident 条件必须同时满足 `index < buffer_size`、`origin == EA` 和年龄窗口，不能依赖物理 ring 顺序。
- B：对年龄 0–29 的 resident EA transition 严格均匀。
- C：年龄 0–4 总质量 0.5，年龄 5–29 总质量 0.5；层内按 transition 均匀。
- 先按原 sampler 产生基准批次，再只 overlay 其中 EA 槽位；因此 EA/RL batch 份额和所有 RL 索引与 A 的基准 draw 一致。
- 每代写入完成后、TD3 更新前构建一次 transition probability/CDF，代内所有更新复用。使用已经审计的 float64 CDF 与 52-bit draw；不要全局打开 JAX x64。

CP100 A 状态下，B 会对 960k 条年龄 0–29 EA transition 均匀，排除 9k（0.93%）age30 transition；其近期质量为 16.67%。C 对 160k recent 和 800k history 各给 50%，因此单条 recent transition 的权重是单条 history transition 的 5 倍。

### 回退与环形安全

- Buffer 尚不能 sample：沿用原 `can_sample`，不执行 overlay。
- 两层都为空：保持 baseline indices 和 intervention key；这是显式安全回退。
- 只有一层非空：该层取得全部 EA 质量，不采样 unset、RL 或已覆盖槽位。
- 建议在 30 代窗口尚未完整前让 B/C 都对当前可见 0–29 代均匀；窗口完整后再启用 C 的 50/50。这避免把 50% 质量压到只有一两个历史代，并使预热规则可预注册、可复现。若从 CP10 分叉，则 C 在 CP30 前与 B 相同。
- 环绕后仍按 sidecar mask 建 CDF；`current_index` 只用于写入，不参与时间层判断。

### JAX/JIT 成本

容量 1M 时，`source_generation` 已存在。新增持久/工作数组约为：float32 probability 4 MB、float64 CDF 8 MB、近期/历史 mask 约 2 MB，合计约 14 MB，另有 JIT 临时量。分布刷新是每代一次 O(capacity)；采样是 EA 槽位上的 `searchsorted`，代内不应重复重建 CDF。无需 source score table、Utility 虚拟更新或 O(source × replay) 矩阵。预期成本显著低于 Utility Replay，但必须在 smoke 中记录 distribution refresh 和 learner sample wall time，不能只凭理论判断。

## 公平 A/B/C Pilot 设计

| 组 | EA 写入 | EA 采样 | RL 槽位 | 作用 |
|---|---|---|---|---|
| A | 全 32 个 | 原始全 resident Uniform | 原始 | 性能基线 |
| B | 全 32 个 | 同 overlay 路径，0–29 代 Uniform | 与基准 draw 逐项相同 | 30 代窗口 + 替换路径对照 |
| C | 全 32 个 | 0–4 代 50% + 5–29 代 50% | 与基准 draw 逐项相同 | 相对 B 隔离时间分层 |

建议：

1. 预先固定一个未参与方法选择的新 seed；从同一个完整 A CP10 状态分叉，三组训练至 CP100。
2. 保持环境、网络、population=32、Buffer 容量、环境交互、TD3 更新、评估日程完全一致；不启用 Fitness admission、Utility 或其他优先级。
3. 主指标预注册为 CP20–100 Return AUC；辅助报告 Return100、首次 Return≥350 的环境步数/实际耗时、总 wall time。
4. 机制审计只记录每组实际 sampled EA 年龄质量（0–4、5–29、≥30）、EA/RL 槽位数、RL index identity、无效索引率、CDF 目标/实测 TV，以及分布刷新和 sampling 额外耗时。
5. B/C 使用相同 overlay 代码和 intervention-key 规则。C 对 B 是时间分层的关键因果对照；A 对 B 同时包含 30 代截断和替换路径/RNG 差异，必须按此限制解释。A 的 CP100 age30 仅占 EA 的 0.93%，所以预计 A/B 干预很弱，但仍需真实结果验证。
6. 不根据中途 Return 调整窗口、50/50 比例、seed 或停止点。

## 决策

**建议进入最小 GPU Pilot（GO），但本阶段不启动。** 理由是：现有数据明确证明硬准入改变了经验时效结构，而当前元数据和已修复 overlay sampler 足以用很小改动直接检验时间分层；同时证据尚不足以支持更复杂的 admission/priority 算法。

进入 Pilot 前的通过条件仅需：目标概率单元测试、ring/warmup 回退测试、RL index identity、无效索引率为 0、checkpoint/resume 的 intervention key 一致性，以及一次短 smoke 的概率忠实度和耗时。若这些条件失败则 HOLD；无需重跑历史大型测试或做参数搜索。

## 新分析产物

- `/disk1/zmh_projects/evorl_artifacts/fitness_experience_admission/seed4099-mechanism-analysis-v1/coverage_analysis.json`
- 复现脚本：`scripts/analyze_fitness_admission_coverage.py`

原 `seed4099-cp100-v1` 目录未写入或覆盖任何文件。
