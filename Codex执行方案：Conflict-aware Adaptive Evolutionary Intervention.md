# Conflict-aware Adaptive Evolutionary Intervention 实施与实验方案

## 一、任务目标

本任务在现有 EvoRL 项目的原始 ERL-TD3 框架上新增一条独立的：

**EA → RL adaptive evolutionary intervention**

通道。

现有 EvoRL ERL 中已有：

```text
RL learner
    ↓
periodic RL injection
    ↓
EA population
```

本任务不能修改其语义。

新增的是：

```text
EA population
    ↓
extract evolutionary direction
    ↓
compare with RL optimization direction
    ↓
Conflict-aware Controller
    ↓
adaptive EA → RL intervention
```

核心研究问题不是：

> EA 与 RL 的方向是否存在冲突。

而是：

> 能否利用在线 EA–RL optimization-direction compatibility，判断 evolutionary information 在什么时候应该进入 RL learner、进入多少，以及冲突究竟是 harmful conflict 还是 potentially useful disagreement。

---

# 二、第一原则：禁止直接修改现有 baseline

本轮研究必须保证原 `ERL-Origin` 可以原样运行。

以下三个文件原则上作为 **read-only baseline reference**：

```text
evorl/algorithms/erl/erl_td3/erl_origin.py
evorl/algorithms/erl/erl_td3/erl_ga.py
evorl/algorithms/erl/erl_td3/erl_td3_workflow.py
```

不要为了实现新算法直接往这三个文件中塞 conflict-aware 逻辑。

新算法通过新增 workflow 实现。

这样最终实验才能明确比较：

```text
Original ERL
vs
Original ERL + Fixed EA→RL
vs
Original ERL + Conflict-aware EA→RL
```

而不是修改 baseline 后失去公平比较对象。

---

# 三、Step 0：必须先建立新分支

任何代码修改之前执行：

```bash
git status --short
git branch --show-current
git log -1 --oneline
```

如果 working tree 不干净：

**禁止执行：**

```bash
git reset --hard
git clean -fd
```

也禁止自动删除、覆盖或 stash 用户已有修改。

必须先报告现有未提交修改并停止代码改动。

如果 working tree clean，则：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git switch -c research/conflict-aware-ea-intervention
```

确认：

```bash
git branch --show-current
```

必须输出：

```text
research/conflict-aware-ea-intervention
```

再开始后续任务。

---

# 四、统一临时验证目录

本任务所有临时文件必须放到：

```text
.codex_validation/conflict_aware_intervention/
```

允许内部建立：

```text
.codex_validation/conflict_aware_intervention/
├── audit/
├── smoke/
├── diagnostic/
├── fixed_transfer/
├── adaptive/
├── ablation/
├── analysis/
└── runs/
```

包括但不限于：

- 临时 Python 脚本
- CSV
- JSON
- debug 日志
- 图片
- checkpoint
- Hydra 输出
- 临时配置
- 参数 dump
- profiling 结果
- 中间分析报告

全部只能进入这个目录。

不要在项目根目录产生：

```text
debug_xxx.py
test_tmp.py
analysis.csv
plot.png
outputs/
multirun/
wandb/
```

验证训练统一使用：

```bash
WANDB_MODE=disabled \
WANDB_DIR=.codex_validation/conflict_aware_intervention/wandb \
PYTHONDONTWRITEBYTECODE=1 \
python scripts/train.py ...
```

同时覆盖 Hydra 输出位置，例如：

```bash
hydra.run.dir=.codex_validation/conflict_aware_intervention/runs/<run_name>
```

运行 pytest 时使用：

```bash
PYTHONDONTWRITEBYTECODE=1 \
pytest -p no:cacheprovider ...
```

防止在仓库留下 `.pytest_cache` 和新的 `__pycache__`。

注意：

正式的：

```text
tests/*.py
configs/*.yaml
evorl/*.py
docs/*.md
```

属于项目代码，不属于临时验证文件，不删除。

---

# 五、Stage 1：代码结构审计，不改变算法

## 目的

先完全确认原 ERL 数据流，不进行任何算法修改。

重点阅读：

```text
evorl/algorithms/erl/erl_td3/erl_origin.py
evorl/algorithms/erl/erl_td3/erl_ga.py
evorl/algorithms/erl/erl_td3/erl_td3_workflow.py
evorl/algorithms/erl/erl_workflow.py
configs/agent/erl/erl-ori.yaml
scripts/train.py
```

必须确认以下数据在一个 iteration 中何时可获得：

```text
pop_actor_params
fitnesses
agent_state before RL update
agent_state after RL update
rl_eval_metrics
ec_eval_metrics
```

并确认：

```text
theta_L_before
    ↓ RL update
theta_L_after
```

可以直接用于构造 RL 实际更新方向。

不要为了获得 gradient 去重构：

```text
agent_gradient_update()
actor_update_fn()
build_erl_rl_update_fn()
```

第一版定义：

\[
g_{RL}
=
\theta_{L,t}^{after}
-
\theta_{L,t}^{before}
\]

这里使用的是 **实际 optimizer-induced update direction**，而不是裸梯度。

它更符合本研究实际要比较的“RL optimizer 真正把 policy 推向哪个方向”。

Stage 1 输出临时审计说明：

```text
.codex_validation/conflict_aware_intervention/audit/code_flow.md
```

该文件最后删除。

### Stage 1 验收条件

必须明确给出：

```text
EC population产生位置
fitness产生位置
RL更新前actor位置
RL更新后actor位置
现有RL→EA injection位置
新增EA→RL intervention最佳插入位置
```

没有搞清楚上述六处之前，不进入 Stage 2。

---

# 六、Stage 2：只实现 Directional Diagnostics

这一阶段：

```text
EA→RL intervention = OFF
```

只采集信号。

不能改变 agent 参数。

---

## 2.1 新建独立工具模块

创建：

```text
evorl/algorithms/erl/erl_td3/conflict_intervention.py
```

这个文件只负责：

```text
PyTree geometry
elite aggregation
direction calculation
controller
parameter intervention
```

第一阶段先实现前四项中的诊断部分。

至少提供这些纯函数：

```python
tree_dot(a, b)
tree_l2_norm(tree)
tree_sub(a, b)
tree_cosine_similarity(a, b, eps=1e-8)

select_elite_centroid(
    pop_actor_params,
    fitnesses,
    topk,
)

population_diversity(
    pop_actor_params,
    eps=1e-8,
)
```

禁止将整棵 actor 参数树 `ravel` 成一个巨大向量后长期保存。

使用：

```text
jax.tree_util.tree_leaves
jnp.vdot
tree_map
tree_reduce
```

直接在 PyTree 上完成计算。

这样避免额外的大规模 GPU 内存复制。

---

# 七、EA direction 的定义

第一版不要只取单一 best individual。

使用 Top-K elite rank-weighted centroid：

\[
\theta_E
=
\sum_{i=1}^{K}
w_i\theta_i
\]

其中：

\[
\sum_iw_i=1
\]

采用 rank weights，而不是 raw fitness softmax。

例如 Top-4：

```text
rank weights = [1, 2, 3, 4]
```

归一化后：

```text
[0.1, 0.2, 0.3, 0.4]
```

best elite 权重最高。

原因：

单一 elite 可能包含大量 mutation noise。

最终：

\[
g_{EA}
=
\theta_E
-
\theta_L^{before}
\]

RL：

\[
g_{RL}
=
\theta_L^{after}
-
\theta_L^{before}
\]

二者使用同一个 anchor：

\[
\theta_L^{before}
\]

因此：

\[
C_t
=
\frac{
\langle g_{RL},g_{EA}\rangle
}{
\|g_{RL}\|
\|g_{EA}\|+\epsilon
}
\]

---

# 八、Stage 2 单元测试

创建正式测试：

```text
tests/test_erl_conflict_intervention.py
```

必须至少覆盖：

### 完全一致

```text
gRL = [1, 0]
gEA = [1, 0]

cos = 1
```

### 完全冲突

```text
gRL = [1, 0]
gEA = [-1, 0]

cos = -1
```

### 正交

```text
gRL = [1, 0]
gEA = [0, 1]

cos = 0
```

### zero direction

任何一个 norm 为 0：

```text
不能 NaN
不能 Inf
```

约定：

```text
cos = 0
```

同时测试：

```text
Top-K elite选择正确
rank weighted centroid正确
population diversity >= 0
```

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
pytest -p no:cacheprovider \
tests/test_erl_conflict_intervention.py -q
```

全部通过再继续。

---

# 九、Stage 3：新增独立 Conflict ERL Workflow

创建：

```text
evorl/algorithms/erl/erl_td3/erl_conflict.py
```

不要修改：

```text
erl_origin.py
```

定义：

```python
class ConflictAwareERLWorkflow(...)
```

或者类似清晰命名。

其整体训练流程必须保持：

```text
1. EC ask
2. EC rollout
3. 获得 fitness
4. RL rollout
5. 保存 theta_L_before
6. RL update
7. 获得 theta_L_after
8. 计算 directional diagnostics
9. optional EA→RL intervention
10. 原始 EC tell / RL→EA injection
11. recorder
```

---

# 十、最重要的时序控制

原始 ERL 还有：

```text
RL → EA
```

注入。

新增方法有：

```text
EA → RL
```

如果处理不当，会形成同 iteration 的立即回流：

```text
EA elite
→ RL
→ 立刻重新注入 EA
```

造成循环污染。

因此必须保存：

```python
agent_state_after_rl
```

然后得到：

```python
agent_state_after_intervention
```

EC 当前 generation 的原始：

```text
RL → EA injection
```

必须使用：

```python
agent_state_after_rl
```

而不是：

```python
agent_state_after_intervention
```

最后写回下一 iteration 的 state 才使用：

```python
agent_state_after_intervention
```

即：

```text
theta_before
    ↓
TD3
    ↓
theta_after_rl ─────────→ 原ERL RL→EA通道
    ↓
Conflict Controller
    ↓
theta_after_intervention
    ↓
next iteration
```

这样本 generation 不会产生：

```text
EA → RL → EA
```

即时 feedback loop。

---

# 十一、Stage 3：增加诊断指标

新 workflow 的 TrainMetric 中加入：

```text
direction_cosine
rl_direction_norm
ea_direction_norm
elite_fitness
rl_return_proxy
ea_advantage_proxy
population_diversity
learner_improvement
intervention_lambda
intervention_mode
```

其中：

\[
A_t^{EA}
=
F_{elite}
-
R_{RL}
\]

第一版明确命名：

```text
ea_advantage_proxy
```

因为 EC fitness 与 RL exploratory rollout 的 return 并非完全严格同分布。

不要把它写成理论上的精确 advantage。

---

# 十二、learner improvement

禁止额外创建新的环境 rollout。

利用当前已经存在的：

```text
rl_eval_metrics.episode_returns
```

构造 RL return EMA：

\[
\bar R_t
=
\beta\bar R_{t-1}
+
(1-\beta)R_t
\]

然后：

\[
I_t
=
\frac{
R_t-\bar R_{t-1}
}{
|\bar R_{t-1}|+1
}
\]

默认：

```yaml
improvement_ema_beta: 0.9
```

第一 iteration：

```text
learner_improvement = 0
```

必须把 EMA state 存在 workflow state/metric 中，而不是 Python mutable global variable。

保证兼容 JAX functional state。

---

# 十三、Population Diversity

第一版定义为 population 到 population centroid 的平均归一化距离：

\[
D_t
=
\frac{
\frac1N
\sum_i
\|\theta_i-\bar\theta\|
}{
\|\bar\theta\|+\epsilon
}
\]

只记录。

**V1 controller 不使用 diversity 决策。**

原因是第一版避免：

```text
C + D + improvement + advantage + uncertainty
```

一下子堆过多变量导致无法消融。

Diversity 留作后续分析和 V2。

---

# 十四、新建配置

创建：

```text
configs/agent/erl/erl-conflict.yaml
```

它必须以：

```text
configs/agent/erl/erl-ori.yaml
```

为参数基线。

只修改：

```yaml
workflow_cls:
```

并增加：

```yaml
conflict_intervention:

  mode: diagnostic

  topk: 4

  start_iter: 20
  interval: 10

  cosine_positive_threshold: 0.2
  cosine_negative_threshold: -0.2

  improvement_ema_beta: 0.9

  fixed_lambda: 0.05

  strong_lambda: 0.05
  soft_lambda: 0.02
  protect_lambda: 0.0
  escape_lambda: 0.02

  max_lambda: 0.05
```

`mode` 支持：

```text
diagnostic
fixed
cosine
conflict_aware
```

默认必须为：

```yaml
mode: diagnostic
```

因此新 workflow 默认不会改变 learner。

---

# 十五、限制第一版范围

Conflict workflow V1 必须明确：

```text
num_rl_agents == 1
```

如果：

```text
num_rl_agents != 1
```

直接给出清晰 assertion/error。

不要第一轮就解决：

```text
multi-RL learner conflict aggregation
```

它属于后续扩展，不属于本研究最小可行性验证。

---

# 十六、Stage 4：必须先验证 diagnostic mode 与 baseline 完全等价

这是整个任务最重要的工程检查之一。

运行：

```text
ERL-Origin
```

以及：

```text
ConflictERL(mode=diagnostic)
```

要求：

```text
同环境
同seed
同population
同训练预算
同全部baseline参数
```

只允许新增日志指标。

算法输出不能被改变。

---

## Smoke Test

首先使用：

```text
Brax Swimmer
```

进行小规模测试。

例如：

```bash
WANDB_MODE=disabled \
WANDB_DIR=.codex_validation/conflict_aware_intervention/wandb \
PYTHONDONTWRITEBYTECODE=1 \
python scripts/train.py \
agent=erl/erl-conflict \
env=brax/swimmer \
seed=0 \
total_episodes=256 \
pop_size=4 \
num_elites=2 \
warmup_iters=1 \
num_eval_envs=8 \
eval_episodes=8 \
conflict_intervention.mode=diagnostic \
hydra.run.dir=.codex_validation/conflict_aware_intervention/smoke/conflict
```

使用等价参数运行：

```text
erl-ori
```

输出：

```text
.codex_validation/conflict_aware_intervention/smoke/baseline/
```

---

## equivalence 必须检查

比较：

```text
eval return sequence
RL return sequence
population return sequence
final actor parameters
sampled episodes
sampled timesteps
```

至少要求：

```python
np.allclose(
    baseline,
    diagnostic,
    rtol=1e-6,
    atol=1e-6,
)
```

如果 diagnostic-only 都改变了结果：

**禁止继续开发 controller。**

必须先找出：

```text
额外PRNG消耗
参数alias
tree mutation
target actor修改
state更新顺序变化
```

等问题。

---

# 十七、Stage 5：正式做 Direction Conflict 诊断实验

这一阶段依然：

```text
intervention OFF
```

目的不是看性能提升。

目的只有一个：

> 判断 optimization-direction conflict 是否在真实 ERL 训练过程中稳定存在，并且是否与之后 learner performance change 有关系。

---

## Pilot 数据集

首先使用：

```text
HalfCheetah
Walker2d
```

每个：

```text
seed = 0, 1, 2
```

先：

```text
10,000 total episodes
```

不要直接跑 100,000。

---

## 分析指标

按 cosine 分为：

```text
Conflict:
C < -0.2

Neutral / complementary:
-0.2 <= C <= 0.2

Aligned:
C > 0.2
```

统计：

```text
各区域占比
平均EA advantage
平均population diversity
平均learner improvement
下一iteration eval return变化
```

重点构造：

\[
\Delta R_{t+1}
=
R^{eval}_{t+1}
-
R^{eval}_{t}
\]

然后分析：

```text
C_t
vs
Delta R_(t+1)
```

---

# 十八、进入算法阶段的 Gate

只有满足以下证据之一，才进入 adaptive controller：

### Gate A

至少两个 seed 中：

```text
C < -0.2
```

事件比例：

```text
>= 10%
```

且 conflict 区域的后续 return change 与 aligned 区域有可观察差异。

或者：

### Gate B

发现明显：

```text
negative cosine
+
learner improving
```

与：

```text
negative cosine
+
learner stagnating
+
EA better
```

对应的后续表现不同。

这说明：

```text
negative cosine
```

不能简单解释为“坏”。

这正好支持：

```text
harmful conflict
vs
useful disagreement
```

这一研究故事。

---

## 如果信号太少

若 10k episodes 不足：

```text
扩展到 30k
```

但仍然只做 diagnostic。

如果：

```text
2 environments
×
3 seeds
×
30k
```

之后：

- cosine 几乎永远一个方向；
- conflict 极少；
- cosine 与后续 performance 完全没有可重复关系；

则：

**停止本方向。**

生成 negative result 总结。

禁止为了让假设成立无限调 threshold。

---

# 十九、Stage 6：实现 Fixed EA→RL baseline

在 adaptive algorithm 之前，必须增加固定 transfer baseline。

否则将来即使 adaptive 方法有效，也无法回答：

> 是 conflict-aware 有效，还是 EA→RL transfer 本身就有效？

在：

```text
conflict_intervention.py
```

加入：

```python
tree_interpolate(
    current_params,
    target_params,
    alpha,
)
```

公式：

\[
\theta_{new}
=
(1-\lambda)\theta_L^{after}
+
\lambda\theta_E
\]

并实现：

```text
mode = fixed
```

默认：

```yaml
fixed_lambda: 0.05
interval: 10
start_iter: 20
```

必须同时调整：

```text
actor_params
target_actor_params
```

朝同一个 elite centroid 做相同系数插值。

禁止修改：

```text
critic_params
target_critic_params
```

第一版也不 reset Adam optimizer state。

---

# 二十、Fixed Transfer 安全测试

测试：

```text
lambda = 0
```

必须严格等价于 diagnostic/baseline。

测试：

```text
lambda = 1
```

actor 必须等于 elite centroid。

测试：

```text
lambda = 0.5
```

必须满足：

\[
\theta_{new}
=
0.5\theta_L
+
0.5\theta_E
\]

并确认：

```text
critic unchanged
optimizer state unchanged
target actor synchronized地进行相同interpolation
```

---

# 二十一、Stage 7：实现 Conflict-aware Controller

在：

```text
conflict_intervention.py
```

加入纯函数：

```python
decide_intervention(
    cosine,
    learner_improvement,
    ea_advantage,
    *,
    positive_threshold,
    negative_threshold,
    strong_lambda,
    soft_lambda,
    protect_lambda,
    escape_lambda,
    max_lambda,
)
```

必须是：

```text
JAX-compatible
JIT-compatible
无Python array branching
```

使用：

```text
jnp.where
jax.lax.cond
```

---

# 二十二、Controller V1 的四种状态

定义 mode id：

```text
0 = NONE
1 = ALIGNED
2 = COMPLEMENTARY
3 = PROTECT
4 = ESCAPE
```

---

## Case 1：Aligned

条件：

\[
C_t>0.2
\]

且：

\[
A_t^{EA}>0
\]

执行：

```text
strong intervention
```

\[
\lambda_t=0.05
\]

解释：

```text
EA 与 RL 方向一致，
且 EA population 提供了更高 return candidate。
```

---

## Case 2：Complementary

条件：

\[
-0.2\le C_t\le0.2
\]

且：

\[
A_t^{EA}>0
\]

执行：

```text
soft intervention
```

\[
\lambda_t=0.02
\]

解释：

```text
EA 提供新的、但没有直接冲突的搜索方向。
```

---

## Case 3：Harmful Conflict / Protect RL

条件：

\[
C_t<-0.2
\]

并且：

\[
I_t>0
\]

执行：

\[
\lambda_t=0
\]

即：

```text
protect learner
```

解释：

```text
RL 自己正在改善，
此时反方向 EA information 不应该强行覆盖 learner。
```

---

## Case 4：Potential Escape

条件：

\[
C_t<-0.2
\]

同时：

\[
I_t\le0
\]

且：

\[
A_t^{EA}>0
\]

执行：

```text
escape intervention
```

\[
\lambda_t=0.02
\]

解释：

```text
RL 已经停滞，
但 evolution 找到了更好的、且方向不同的 candidate，
这种 disagreement 可能恰恰是 EA 的探索价值。
```

---

## 其他情况

全部：

\[
\lambda_t=0
\]

---

# 二十三、为什么不能简单 C<0 reject

不要实现：

```python
if cosine < 0:
    lambda = 0
```

因为这会破坏本研究最重要的问题：

```text
harmful conflict
vs
useful disagreement
```

EA 的一个重要价值就是离开 gradient learner 当前搜索方向。

所以：

```text
negative cosine
```

本身不能直接等价于：

```text
bad evolutionary direction
```

---

# 二十四、Controller 单元测试

必须人工构造：

### Aligned

```text
C = +0.8
I = +0.1
A = +0.3
```

结果：

```text
mode = ALIGNED
lambda = strong_lambda
```

### Complementary

```text
C = 0.0
A = +0.3
```

结果：

```text
COMPLEMENTARY
soft_lambda
```

### Protect

```text
C = -0.8
I = +0.1
A = +0.3
```

结果：

```text
PROTECT
lambda = 0
```

### Escape

```text
C = -0.8
I = -0.1
A = +0.3
```

结果：

```text
ESCAPE
lambda = escape_lambda
```

### Bad EA

```text
C = -0.8
I = -0.1
A = -0.3
```

结果：

```text
NONE
lambda = 0
```

---

# 二十五、Stage 8：第一轮真正的算法实验

选择 Stage 5 中 directional signal 最明显的一个环境。

固定：

```text
seed = 0,1,2
相同training budget
相同population
相同TD3参数
相同evaluation
```

运行四组：

```text
B0 Original ERL

B1 ERL + Fixed EA→RL

B2 ERL + Cosine-only EA→RL

B3 ERL + Conflict-aware EA→RL
```

其中 B2：

```text
C > 0.2 → strong
otherwise → no transfer
```

它是很重要的消融。

因为需要证明：

```text
仅仅cosine gating
```

不等同于：

```text
distinguishing harmful conflict and useful disagreement
```

---

# 二十六、第一轮实验必须记录

除了最终 return，还必须记录：

```text
mean return
best return
learning curve
sample efficiency
direction cosine distribution

aligned rate
complementary rate
protect rate
escape rate

mean lambda
number of interventions

return before intervention
return after intervention

EA advantage distribution
learner improvement distribution
population diversity
```

同时统计：

```text
training wall-clock
time_cost_per_iter
```

确认新机制本身没有产生巨大计算成本。

---

# 二十七、算法继续推进 Gate

B3 进入正式大规模实验前，应至少满足：

### 条件 1

Controller 确实触发多种状态。

不能出现：

```text
99% ALIGNED
```

或者：

```text
99% NONE
```

否则 adaptive controller 实际已经退化。

### 条件 2

B3 与 B1 Fixed Transfer 有可观察差异。

否则说明：

```text
adaptive signal没有真正提供价值。
```

### 条件 3

至少两个 seed 不劣于 Original ERL，且整体趋势不是由单 seed 异常造成。

### 条件 4

`lambda=0` 时完全退化为 baseline。

---

# 二十八、Stage 9：只有通过 Pilot 后才能扩大实验

如果 Stage 8 成立，再进入：

```text
HalfCheetah
Hopper
Walker2d
Ant
```

正式实验建议：

```text
5 seeds
```

最终训练预算恢复与 `erl-ori.yaml` baseline 相同，例如：

```text
100000 total episodes
```

所有方法完全相同预算。

至少比较：

```text
ERL-Origin
Fixed EA→RL
Cosine-only
Conflict-aware
```

必要消融：

```text
w/o learner improvement
w/o EA advantage
w/o escape state
```

不要一开始跑十几个变体。

先证明主方法成立再消融。

---

# 二十九、什么时候允许优化代码

禁止在 Diagnostic 阶段提前大规模重构。

只有出现以下情况才优化：

```text
Conflict workflow time/iter
>
Original ERL time/iter × 1.08
```

即额外 overhead 超过约 8%。

如果超过：

优先优化：

```text
evorl/algorithms/erl/erl_td3/conflict_intervention.py
```

检查：

```text
是否重复flatten parameter
是否产生大temporary vector
是否重复计算norm
是否没有JIT
是否重复tree traversal
```

其次优化：

```text
erl_conflict.py
```

把：

```text
direction calculation
controller decision
interpolation
```

组合成 JIT-friendly pure function。

**不要为了性能首先修改 TD3 核心代码。**

---

# 三十、什么时候允许重构

只有以下阶段全部通过：

```text
unit tests
diagnostic equivalence
signal pilot
fixed transfer smoke
adaptive smoke
```

之后才能重构。

重构主要针对：

```text
erl_conflict.py
conflict_intervention.py
```

目标是把 workflow 中的研究逻辑收敛成：

```python
diagnostics = compute_conflict_diagnostics(...)

decision = decide_intervention(...)

agent_state = apply_intervention(...)
```

保持：

```text
step()
```

只负责训练时序。

不要把大量公式直接散落在 `step()` 中。

---

# 三十一、除非绝对必要，不修改这三个 baseline 文件

```text
erl_origin.py
erl_ga.py
erl_td3_workflow.py
```

如果 Codex 判断确实必须修改其中某个文件：

在修改前必须证明：

1. 无法通过 subclass/helper 实现；
2. 修改不会改变 ERL-Origin 默认行为；
3. 有对应 regression test；
4. `erl-ori` 同 seed short-run 仍保持等价。

否则不要修改。

---

# 三十二、最终项目中应该留下的文件

理想情况下正式新增仅包括：

```text
evorl/algorithms/erl/erl_td3/
    conflict_intervention.py
    erl_conflict.py

configs/agent/erl/
    erl-conflict.yaml

tests/
    test_erl_conflict_intervention.py

docs/research/
    conflict_aware_intervention.md
```

如果项目当前不存在：

```text
docs/research/
```

允许建立。

---

# 三十三、最终研究报告

创建正式文档：

```text
docs/research/conflict_aware_intervention.md
```

必须写清：

```text
1. Research Question

2. Original EvoRL ERL architecture

3. 原代码只有 RL→EA injection 的事实

4. 新增 EA→RL intervention 的原因

5. gRL 定义

6. gEA 定义

7. cosine compatibility 定义

8. learner improvement

9. EA advantage proxy

10. harmful conflict / useful disagreement 四状态定义

11. 代码改动位置

12. Diagnostic experiment

13. Conflict frequency

14. Fixed-transfer experiment

15. Adaptive experiment

16. Ablation

17. 各seed完整结果

18. Mean ± Std

19. Wall-clock overhead

20. 结论

21. 当前局限

22. 是否值得继续正式论文实验
```

不能只写：

```text
“方法有效”
```

必须给数字和证据。

如果失败，也要如实写：

```text
hypothesis unsupported
```

以及为什么关闭该方向。

---

# 三十四、临时文件最终必须统一删除

当正式 report 已经吸收所有需要保留的实验结果后：

首先确认：

```text
docs/research/conflict_aware_intervention.md
```

已经包含最终结果。

再删除：

```bash
rm -rf .codex_validation/conflict_aware_intervention
```

然后确认：

```bash
test ! -e .codex_validation/conflict_aware_intervention
```

禁止使用：

```bash
git clean -fd
```

因为可能误删用户其他未跟踪文件。

---

# 三十五、最终污染检查

执行：

```bash
git status --short
```

再检查：

```bash
find . -maxdepth 2 \
  \( \
    -name "debug_*" \
    -o -name "tmp_*" \
    -o -name "*.tmp" \
    -o -name ".pytest_cache" \
  \) -print
```

以及：

```bash
git ls-files | grep ".codex_validation" || true
```

必须保证：

```text
.codex_validation不存在
没有debug脚本
没有临时CSV
没有临时plots
没有Hydra验证输出
没有validation checkpoint
```

项目中只保留真正应该长期维护的：

```text
source code
config
tests
research report
```

---

# 三十六、最终测试

至少运行：

```bash
PYTHONDONTWRITEBYTECODE=1 \
pytest -p no:cacheprovider \
tests/test_erl_conflict_intervention.py -q
```

然后运行项目中与 ERL / TD3 相关的现有测试。

最后重新执行一个极短：

```text
erl-ori
```

和：

```text
erl-conflict mode=diagnostic
```

regression smoke。

确认：

```text
Original ERL没有被破坏
Conflict diagnostic等价baseline
Conflict-aware workflow能够独立运行
```

---

# 三十七、Git 提交策略

不要一次提交整个实验。

至少分成以下提交：

```text
Commit 1
test/feat: add ERL directional geometry utilities

Commit 2
feat: add non-invasive ERL conflict diagnostics

Commit 3
feat: add EA-to-RL intervention baseline

Commit 4
feat: add conflict-aware adaptive intervention

Commit 5
test: add conflict-aware ERL regression coverage

Commit 6
docs: summarize conflict-aware intervention experiments
```

每次 commit 前：

```bash
git diff
git status --short
```

确认没有：

```text
checkpoint
wandb
outputs
CSV
PNG
debug script
.codex_validation
```

被加入 commit。

本任务暂不自动 push，除非收到明确 push 指令。

---

# 三十八、整个任务的停止条件

Codex 不允许为了得到“好结果”无休止调参。

### Close Condition A：假设不成立

如果：

```text
2 env
×
3 seeds
×
足够训练阶段
```

显示：

```text
direction conflict基本不存在
```

或者：

```text
cosine与后续performance没有稳定关系
```

则关闭该方向。

---

### Close Condition B：transfer 本身有害

若：

```text
lambda = 0.05
lambda = 0.02
lambda = 0.01
```

均稳定严重破坏 learner，

则不要继续设计复杂 controller。

优先分析：

```text
parameter-space distance
policy functional mismatch
elite noise
target-network inconsistency
```

---

### Close Condition C：adaptive 没有比 fixed 更有价值

如果：

```text
Fixed EA→RL
≈
Conflict-aware EA→RL
```

且多个 seed 都如此，

则不能声称 conflict-aware controller 有贡献。

---

### Continue Condition

只有当：

```text
directional conflict确实存在
+
conflict状态与后续表现存在关系
+
adaptive intervention优于/稳定于fixed transfer
```

三项形成完整证据链后，

才扩大到正式多环境、多 seed 实验。

---

# 三十九、Codex 最终必须返回的内容

任务完成后，不要只回复“已完成”。

最终回复必须给出：

```text
1. 当前branch

2. commit列表

3. 新增文件

4. 修改文件

5. 明确说明是否修改baseline文件

6. 原ERL数据流

7. 新数据流

8. Diagnostic结果

9. Conflict出现频率

10. Fixed transfer结果

11. Adaptive结果

12. 各seed结果

13. Ablation结果

14. Performance overhead

15. 哪些实验通过

16. 哪些实验失败

17. 当前研究结论

18. 是否满足继续正式实验的gate

19. .codex_validation是否已经删除

20. 最终git status
```

最后单独给出：

```text
RESEARCH STATUS:
CONTINUE
```

或者：

```text
RESEARCH STATUS:
CLOSE
```

不得因为结果不好而自动选择 CONTINUE。

判断必须由数据决定。