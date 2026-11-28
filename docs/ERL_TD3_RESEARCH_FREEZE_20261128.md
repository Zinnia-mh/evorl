# EvoRL-TD3 研究阶段封版

封版日期为 **2026-11-28（北京时间）**。本阶段结束。各实验的源码、分支、日志、配置与 checkpoint 保留在原位置。本提交以 Temporal Replay 报告提交 `9ad7b878f167bbe4c9d55e9678f5a7853105ec1b` 为父提交，仅新增本总结。各机制应按索引中的独立源码版本读取，归档树无需包含全部历史分支的算法。

## 研究问题

ERL-TD3 的 EA 个体用回报衡量 Fitness，生成的轨迹进入共享 Replay。研究首先检验 Fitness 能否代表经验对当前 learner 的价值，然后测试按来源打分、按 Fitness 准入、按时间分层等干预能否提升等环境预算下的回报 AUC。Learner Utility 定义为真实 TD3 Critic 与 Optax 状态上的一次虚拟更新所带来的冻结参考批次损失下降，$U=L_{ref,before}-L_{ref,after}$。它量化局部 Critic 更新效果，长期样本效率和计算效率由独立训练评价。见[诊断定义与验收](https://github.com/Zinnia-mh/evorl/blob/df954935334e201f3e2759b848d5be2c177833f7/docs/learner_utility_diagnostic/FINAL_REPORT.md)。

## 诊断结论

- **Fitness 与 Utility 弱正相关。** Swimmer seeds 1/2/3，各 302 个 checkpoint、每个 32 个 EA 来源，共 28,992 条来源记录。checkpoint 平均 Spearman 为 0.2443，排序反转率 41.76%，Top-F 与 Top-U 一致率 7.17%，负 Utility 比例 88.98%。这是当前诊断样本和参考分布下的观察。见[三 seed 诊断报告](https://github.com/Zinnia-mh/evorl/blob/df954935334e201f3e2759b848d5be2c177833f7/docs/learner_utility_diagnostic/FINAL_REPORT.md)。
- **来源、训练阶段和虚拟步长影响分布。** 来源对照每个 seed 50 个有效 checkpoint。步长 1.0、按固定数值容差判定时，EA、RL-own、Replay-random、Reference-self 的三 seed 平均负率依次为 90.9%、74.8%、50.4%、10.5%。早、中、晚三段 Fitness–Utility 平均 Spearman 为 0.294、0.255、0.237。步长 0.1、0.5、1.0 的分布、正负与参考一致性保存在来源对照的 CSV 和图表，归因分类仍为 AMBIGUOUS。见索引 L2 的 `visualization_report.md` 和 `tables/`。
- **固定 learner 的条件重复性成立，跨时间保序证据有限。** 用 seed2047 CP50/300 的封存 learner 和驻留经验，每个 checkpoint 128 个来源、每个条件 8 次重复。改变独立参考批次时，cohort 内 Spearman 为 0.861–0.892，仍有约 12.3%–14.5% 配对反转。历史评分与当前 learner 重测在 10–30 代间的保序性较弱，且历史 minibatch 存档有限。见索引 L3 的 `REPORT.md` 及[历史评分分析](https://github.com/Zinnia-mh/evorl/blob/6d5b8578fb22f144951327ece9f2c025b052cb2c/docs/budgeted_utility_pilot.md)。
- **其他来源假设。** RL-derived 历史诊断以 seed2027 CP50/150/300 的固定状态比较共享扰动方向的派生策略，未见相对 EA 或未扰动策略的一致 Utility 优势。在线 M、M-Late 在 seed3083 分别较 A 的 AUC60–150 下降 25.35%、AUC100–150 增加 0.41%，评价窗口不同。另行完成的全局 EA–RL 方向余弦诊断未支持稳定的自适应控制信号。见索引 E1–E3。

## 已实现并验证的机制

- **Utility 到真实 TD3 输入。** 用带来源标记的驻留经验和真实虚拟 Critic 更新打分，在资格及有效期约束下生成来源权重并映射到 Replay 索引。先取原生 Uniform 批次，独立干预随机流只重抽 EA 槽位。D0、S/P 的在线审计保存了真实 Critic 输入证据，确认 EA 索引改变、RL 位置和索引保持一致。D1 独立开发实验的指定端点使用确定性重建证据。相关输入、权重及审计在索引 R2–R7。重抽索引比例和真正重分配的概率质量 TVD 是两项独立指标。
- **数值忠实性。** 对固定 CP50/300 历史分布，CDF 局部高精度及 52-bit 随机网格将来源概率 L1 误差由约 0.0157 降到约 $4\times10^{-9}$。原实现的训练结果继续作为原版本记录。见[采样器精度报告](https://github.com/Zinnia-mh/evorl/blob/404a01b3d796967656d8b9309d3c460a4c24301d/docs/utility_replay_sampler_fidelity.md)。
- **Fitness 准入。** A 每代完整写入 32 条 EA 轨迹。Top16 或 Random16 均完成 32 个个体的评估，但只写入 16 条 EA 轨迹。Top16 每代精确采用 Fitness 排序，Random16 独立抽取，二者均沿用原 Uniform learner 采样并保留 RL 写入。seed4099 CP100 的 EA 驻留均龄从 A 的 14.64 代升至两种半准入的 28.91 代，EA/RL 驻留比例由 96.9/3.1% 变为 94.1/5.9%。半准入与 A 的比较同时包含容量覆盖与年龄结构变化。见[覆盖与时效分析](https://github.com/Zinnia-mh/evorl/blob/9b46e32c808e6756816555ab204f1caf6b208231/docs/fitness_admission_mechanism_diagnosis.md)及 F1–F2 的 `verification.json`。
- **时间分层。** A 为全驻留 Uniform，B 为最近 30 代内均匀，C 对年龄 0–4 与 5–29 分别分配 50% EA 概率。B/C 从 CP30 激活，实测两层质量分别为 16.6671/83.3329% 和 49.9782/50.0218%。各组每代写入 32,000 条 EA 和 1,000 条 RL，B/C 的活跃审计无 RL 索引变更或越龄 EA 索引。CP30→31 Smoke 验证了环形覆盖和采样随机流恢复。见[Temporal 实验报告](https://github.com/Zinnia-mh/evorl/blob/9ad7b878f167bbe4c9d55e9678f5a7853105ec1b/docs/temporal_stratified_replay_seed5119_cp100.md)及 T1–T2。

## 性能观察与当前证据边界

| 比较 | 实测结果 | 封版判断 |
|---|---|---|
| A 与稀疏 D0 | seed1 CP20–3031，D0 对 A 的 AUC +0.70%，耗时 +60.28%。seed2047 CP300 匹配复验 AUC −12.20%，训练调用 2.19h 对 1.37h。共同 CP19 的 seeds 2111/2129 中，D0 对置换 P 的 CP20–120 AUC 分别 −20.01%/−21.71%。 | 当前 D0 真实 Utility 排名的增量效益缺乏支持。历史 seeds 2081/2099 的 D0/P 符号相反，且起点证据不同。出处 R1–R3。 |
| D1 开发对照 | seed2027 CP20–300，D1 对 A 的 AUC +13.14%，对 D0 −1.04%。A/D0/D1 进程耗时 1.44/2.45/4.86h。 | 当前单 seed 开发预算未支持 D1 优于 D0。出处 R4。 |
| S 预算调度与同调度 P | seed3079 CP20–50，S 对 A +24.54%、对 D0 +59.28%，优势集中于末点评估。CP20–150，S 对 A 在 seeds 3079/3083 分别 +91.85%/+16.21%，S 对 P 则为 +9.22%/−23.17%。同步训练 step 总耗时约为 A 的 2.6 倍。 | 同调度真实排名的收益未跨 seed 复现。环境步 AUC 和墙钟收益需各自评价。出处 R5–R7。 |
| R 随机来源优先级与 U-Overlay | 不计算 Utility 的 R 在两 seed CP20–150 对 A 的 AUC 分别 +51.04%（3079）、+8.50%（3083），对 P 为 −14.02%/−28.27%，用时约为 P 的一半。均匀 EA overlay 的 U 在 CP100 对 A 的 AUC，3079 为 510.154M 对 271.393M，3083 为 208.315M 对 294.868M。 | R 未达到与 P 接近的预设门槛。U 结果依 seed 而变，采样路径与随机流效应仍未单独识别。出处 R8–R9。 |
| Fitness Top16 与 Random16、A | CP20–100，seeds 3083/3079/4099 的 Top16 对 A 为 +103.63%/−5.60%/−25.41%，对 Random16 为 +132.34%/+27.21%/−6.34%。 | Top16 的性能排序未跨 seed 稳定复现。出处 F1–F2。 |
| 固定 5+25 Temporal | seed5119 CP20–100 的 A/B/C AUC 为 639.533M/465.999M/385.465M。C 对 B −17.28%，对 A −39.73%。CP100 回报 366.51/355.47/368.20，墙钟 28.41/30.26/30.87 min。 | 当前固定配置 STOP。单 seed 结果保留为观察，其他时间分层方案仍未被检验。出处 T2–T3。 |

CP11 六次独立 GPU 进程从同一个封存 CP10 恢复，状态和 PRNG 一致，分层尚未激活。A/A、B/B、C/C 的 32 个 EA 个体回报分别有 32、26、32 个不同。原始轨迹首帧观测一致，第一次环境推进后的奖励和观测已出现差异，随后 Replay 内容和首批 TD3 payload 分化。采样索引是按原函数和真实入口 key 重建的，配对保持一致。单独的物理前缀探针在部分比较中还观察到 reset 内部量的微小数值差异。证据支持此环境下 GPU rollout 的跨进程数值可重复性有限，具体算子未确认。seed5119 原始 AUC 与机制审计仍有效，单 seed 性能差异的归因受到限制。见 T3 的 `comparison.json` 与六份捕获。

主要局限是多数干预实验集中在 Brax/Swimmer 和少数开发 seed，checkpoint 之间存在时间依赖，局部参考批次 Utility 无法直接推出长期策略回报。同一算法在不同预算、阈值与计时口径下的结果应保持原区间。现有证据验证了采样与准入机制，尚不足以支持这些干预在独立 seed 上带来稳定样本效率或计算效率增益。

## 实验与恢复索引

下表路径相对 4090 服务器上的根目录。**A** 为 `/disk1/zmh_projects/evorl_artifacts`。**S** 为 `/disk1/zmh_projects/evorl_utility_source_controls_outputs`。**P** 为 A 下的 `learner_utility_replay_pilot`。**Q** 为 A 下的 `learner_utility_signal_study`。报告文件相对于各行产物目录，Git 报告用固定版本链接。

| 编号与实验 | 版本 | Seed、进度与 checkpoint | 远程产物及报告 |
|---|---|---|---|
| L1 Learner Utility | `research/learner-utility-diagnostic@df95495`，诊断运行 `76309c5` | 1/2/3，完成 3×302 个诊断点。三个正式目录的 `checkpoints/` 无可用 payload，原始记录和配置可重分析 | A/`learner_utility_diagnostic`，[报告及清单](https://github.com/Zinnia-mh/evorl/tree/df954935334e201f3e2759b848d5be2c177833f7/docs/learner_utility_diagnostic) |
| L2 EA/RL-own/Replay-random/Reference-self | `research/utility-source-controls@8e551ae` 为分支封存头，运行版本见各 `run_config.json` | 1/2/3 正式来源对照完成，42 为 Smoke。主要产物为观察记录，未发现正式目录保存可恢复模型 | S/`formal_v1`，`visualization/visualization_report.md`、`seed_*/artifacts/diagnostic_report.md`，[协议](https://github.com/Zinnia-mh/evorl/blob/8e551ae36e4d629b47f1a501df74306dccd6de0e/docs/learner_utility_source_controls/README.md) |
| L3 排名重复性 | `codex/d0-utility-ranking-repeatability@85d07f6` | 2047，完成 CP50/300 冻结分析，原 learner checkpoint 见 R2 | P/`analysis/utility-ranking-repeatability-v1`，`REPORT.md`、`postrun_audit.json`、`checksums.sha256` |
| R1 A/B/C/D0 基线与来源评分对照 | `codex/archive/learner-utility-replay-d0-seed1-v1@228a301`，标签 `learner-utility-replay-d0-v1-seed1` | seed1 四臂完成并保留 CP3031。seeds 2/3 同根清单为协议，勿计入已完成的正式重复 | P/`formal/swimmer/abc-baseline-v1-head228a301`，`analysis-seed1-v1/summary.json`、`release-records/d0-v1-seed1/` |
| R2 A/D0 与采样器忠实性 | `experiment/d0-utility-placebo@43221c7`；修复 `codex/d0-sampler-fidelity@404a01b` | 2047，A/D0 CP300 完成并保留。CP50/300 有历史输入供只读验证 | P/`replication/swimmer/seed_2047/ad300-head43221c7-v1` 的 `REPORT.md`、`sampling_verification.json`；P/`analysis/ad2047-weak-intervention-v1`、P/`analysis/d0-sampler-fidelity-v1`，[修复报告](https://github.com/Zinnia-mh/evorl/blob/404a01b3d796967656d8b9309d3c460a4c24301d/docs/utility_replay_sampler_fidelity.md) |
| R3 D0 置换与共同起点 | `experiment/d0-utility-placebo@43221c7`，历史 `b5fb4eb` | 2081/2099 历史 CP120，部分早期 payload 已轮换。2111/2129 共同 CP19 至 CP120 完成，永久父状态与保留的子 checkpoint 可按清单定位 | Q/`d0-placebo-ablation-v1`、`phase14-shared-start-v1`、`phase15-resampling-audit-v1`，Q/`phase15-d0-evidence-closeout-v1/EVIDENCE_SUMMARY.md`、`PROVENANCE.json`、`ARCHIVE_INVENTORY.csv` |
| R4 D1 开发 | `codex/archive/learner-utility-replay-d1-300-v1@6b771f2`，标签 `learner-utility-replay-d1-v1-dev300` | 2027，D1 CP300 完成有封存点，历史 HOLD 日志保留 | P/`d1/development/swimmer/seed_2027/d1-300-v1-head6b771f2`，`analysis/d1-5-v4/README.md`、`sealing/d1-v1-dev300/verification.json` |
| R5 A/D0/S 预算调度 | `codex/d0-budgeted-scoring@6d5b857`，运行 `535c6f0` | 3079，三臂 CP50 完成且有封存点。成功版与早期 Smoke 尝试各自留档 | P/`algorithm/budgeted-scoring-v2/pilot`，`REPORT.md`、`completion_verification.json`，[Git 报告](https://github.com/Zinnia-mh/evorl/blob/6d5b8578fb22f144951327ece9f2c025b052cb2c/docs/budgeted_utility_pilot.md) |
| R6 A/S/P 延长 | `codex/s-utility-placebo-cp150@ba5c570`，运行 `029acf3` | 3079，A/S 从 CP50、P 从共同 CP10 继续，均保留 CP150 | P/`algorithm/budgeted-scoring-cp150-v1`，`REPORT.md`、`recovery_checks.json`、`runs/*/segments/*/critic-input-*.npz`，[Git 报告](https://github.com/Zinnia-mh/evorl/blob/ba5c5708703a2ef30b8399157a87655ca1bb2d1c/docs/s_utility_cp150.md) |
| R7 A/S/P 复验 | `codex/s-utility-replication-seed3083@f1cdb57`，运行 `1aafee9` | 3083，三臂 CP150 完成，保留 CP10/50/100/150 | P/`algorithm/s-utility-replication-seed3083-v1`，`REPORT.md`、`comparison.json`、`runs/*/segments/*/critic-input-*.npz`，[Git 报告](https://github.com/Zinnia-mh/evorl/blob/f1cdb57b63cfda9c843a4a82acd163ead1df7fa3/docs/s_utility_seed3083_results.md) |
| R8 R 随机来源优先级 | `codex/random-source-priority@7693701` | 3079/3083，R CP150 均完成且有封存点。A/P 见 R6–R7 | P/`algorithm/random-source-priority-v1`，`REPORT.md`、`final-verification.json`，[Git 报告](https://github.com/Zinnia-mh/evorl/blob/76937014bda6e2bf86e265c2558b3d75260f5020/docs/random_source_priority_results.md) |
| R9 U-Overlay | `codex/uniform-replay-overlay@976b279` | 3079/3083，U CP100 均完成且有封存点。A/P/R 对照取相同窗口 | P/`algorithm/uniform-overlay-cp100-v1`，`REPORT.md`、`frozen-rng-verification.json`，[Git 报告](https://github.com/Zinnia-mh/evorl/blob/976b27970dd58dafcc690a981a1d3ac877ae84f7/docs/uniform_overlay_cp100.md) |
| F1–F2 Fitness Admission | `codex/fitness-experience-admission@9b46e32c808e6756816555ab204f1caf6b208231` | 3079/3083 的 Top16、Random16 与已有 A；4099 三臂。均完成 CP100。4099 三臂保留 CP10/25/50/75/100 并有 PASS 记录 | A/`fitness_experience_admission/{cp100-v1,seed4099-cp100-v1,seed4099-mechanism-analysis-v1}`，[双 seed 报告](https://github.com/Zinnia-mh/evorl/blob/9b46e32c808e6756816555ab204f1caf6b208231/docs/fitness_experience_admission_cp100.md)、[4099 报告](https://github.com/Zinnia-mh/evorl/blob/9b46e32c808e6756816555ab204f1caf6b208231/docs/fitness_experience_admission_seed4099.md) |
| T1 Temporal Smoke | `codex/temporal-stratified-replay@19f0f0eadc931bec76654f5f4f3bfd4243b712d4` | 5119，A/B/C CP31 完成，B 的 CP30/31 存在两个 segment | A/`temporal_stratified_replay/smoke-seed5119-cp31-v1`，`smoke.log`、`comparison.json`、`runs/*/completion.json` |
| T2 Temporal Pilot | 实现 `19f0f0e`，`codex/temporal-stratified-replay@9ad7b878f167bbe4c9d55e9678f5a7853105ec1b` | 5119，A/B/C CP100 完成，各有 CP10/25/50/75/100。既有只读核查成功恢复 CP10/25/100 | A/`temporal_stratified_replay/seed5119-cp100-v1`，`comparison.json`、`analysis.json`、`learning_curves.csv`，[Git 报告](https://github.com/Zinnia-mh/evorl/blob/9ad7b878f167bbe4c9d55e9678f5a7853105ec1b/docs/temporal_stratified_replay_seed5119_cp100.md) |
| T3 CP11 复现 | T2 源码与同一份原 A CP10，探针脚本独立留存 | 5119，A/B/C 各两次 GPU 独立进程，至首次 TD3 更新前。无新训练 checkpoint，恢复入口见 T2 | A/`temporal_stratified_replay/cp11-repro-seed5119-20261128-v1`，`comparison.json`、`driver.log`、`probe.py`、六份 `.npz` |
| E1 RL-derived 来源诊断 | `research/rl-derived-ea-utility@c6b68de` | 2027，完成基于既有 A CP50/150/300 的三点诊断，沿用历史 checkpoint | S/`mechanism/shared_epsilon_historical_20261124_v2`，`mechanism_report.md`；早期 Smoke 见 S/`smoke` |
| E2 在线 M / M-Late | `codex/rl-guided-population-pilot@1f16fb2`、`codex/rl-guided-late-pilot@8e5a898` | 3083，均完成 CP150 并保留封存点 | P/`algorithm/{rl-guided-population-seed3083-v1,rl-guided-late-seed3083-v1}`，各 `REPORT.md` 与 `utility_summary.json` |
| E3 方向余弦诊断 | 本地 `research/conflict-aware-ea-intervention@636b4494cf368eb4efb46c3cda83f76182599ed0` | HalfCheetah/Walker2d，10k 与各三条 30k fresh-seed 轨迹完成。此条以原报告为索引，未查验续训 checkpoint | 主仓库用 `git show 636b449:docs/research/conflict_aware_directional_diagnostics_closure.md` 读取 |

## 封存核对与恢复注意事项

- 封版前六个已登记的 worktree 均干净。主 worktree `/disk1/zmh_projects/evorl` 位于 `research/rl-derived-ea-utility@c6b68de2a5ee3899003e3e2273e9e75a25144bea`，与实查 origin 一致。Fitness、Temporal、来源对照及 Uniform overlay 的 checkout 同样对应现有远端头。
- 本地 `research/learner-utility-replay-pilot` 为 `228a3014902948251998bc8a39cad128c910ec6e`，同名远端为 `3631a02498901fbbef8945fec9266ffa5c7344b1`。本地 head 另有远端归档分支与 seed1 tag。实验分支 `experiment/d0-utility-placebo` 没有同名远端 ref，其 `43221c7` 已由远端多个后继分支保留。E3 的方向余弦分支仅确认了远程服务器本地 ref。各历史 ref 原位保留。
- 封版检查时没有 EvoRL 训练进程。服务器留有 `evorl-utility:review` 的空闲 shell，以及 seed2027 历史 `ad300-v1-head053e9de` 的 tail/awk 监控。该实验的 driver 日志记录 `EXIT=130` 和 `HOLD`。相关进程与产物原样保留。
- 核查采用已有 seal 和验证记录。对 Fitness、Temporal、主 Replay pilot 的完成产物，检查了 `completion.json` 所指的 checkpoint、`state/`、`runtime/metadata`、seal 列出的文件及小体积 `report.json` 哈希。大型 payload 沿用既有完整性校验。因原轮换缺失的历史节点和仅保存记录的诊断已在索引注明。Git 保存代码和证据索引，实验 payload 仍位于上述远程目录。
- 若日后获准恢复，先按具体实验的运行 commit、原 resolved config、原环境合同和既有 seal 只读验证目标 checkpoint，然后另建输出目录。完整继承 learner、optimizer、GA、Replay 写指针及元数据、训练与评估 PRNG。已完成目录受 manifest 和目标预算约束，扩展需新合同，避免在恢复边界重复评价并提前消耗 key。对跨进程 GPU 运行，应把共享 checkpoint 与轨迹逐项复现作为两种独立检查，并保留失败/中断的原始记录。
