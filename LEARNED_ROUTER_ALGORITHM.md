# MathChecker 当前算法完整流程说明

本文档说明当前项目中已经实现的完整算法流程，重点覆盖以下内容：

- 整体目标是什么
- 推理阶段每一层在做什么
- `hybrid step type classifier` 和 `learned router` 分别承担什么职责
- `specialist review` 如何参与最终判断
- `learned router` 的训练数据如何构造
- 如何训练 `Qwen3-0.6B` 路由模型
- 如何在推理时接入训练好的路由模型

本文档描述的是当前仓库中已经落地的版本，而不是纯方案设计稿。

## 1. 任务目标

这个项目要解决的问题是：

- 输入一道数学题
- 输入模型生成的多步解题过程
- 对每一步做局部判别
- 找出“第一处错误”出现的位置

项目不是直接判断最终答案对不对，而是判断：

- 当前步骤是否与题目条件兼容
- 当前步骤是否与前序步骤兼容
- 当前步骤是否存在算术、代换、等价变形、条件义务等层面的冲突

最终目标是提高：

- 第一处错误定位准确率
- 多步推理轨迹上的步级判别稳定性
- 对“不同但正确的替代解法”的容错能力

## 2. 当前系统的整体结构

当前系统可以分成 7 个层次：

1. `stage1`
2. `stage2`
3. `hybrid step type classifier`
4. `learned router`
5. `stage2_review`
6. `stage2_specialist_review`
7. deterministic specialist fallback adjustment

其中：

- `stage1` 负责抽取“下一步应该用到的数学概念、分析、表达式”
- `stage2` 负责对当前实际步骤做三维标签判断
- `step type classifier` 负责给当前步骤做解释性分类
- `learned router` 负责直接选择 specialist 组合
- `stage2_review` 是通用复核层
- `stage2_specialist_review` 是带 specialist evidence 的复核层
- deterministic adjustment 是最后的保底纠偏层

## 3. 推理阶段的完整流程

### 3.1 输入

对于一个样本，系统输入包括：

- `question`
- `steps`
- `gold_answer`
- `model_answer`
- `gold_first_mistake_index`（评测时可用，不参与普通推理）

系统按步骤依次处理，每次聚焦一个 `current_step`。

### 3.2 Stage 1：生成“下一步应关注的信息”

`stage1` 的目标不是判错，而是从题目和前序步骤中抽取出：

- `Mathematical Concepts to Apply`
- `Key Analyses for the Next Step`
- `Mathematical Expressions to Compute`

这一步的作用是给后续 `stage2` 一个软参考，而不是唯一正确路径。

### 3.3 Stage 2：对当前步骤做三维局部判别

`stage2` 会围绕当前实际步骤输出三个维度的标签：

- `mathematical_concepts`
- `key_analyses`
- `calculations`

每个维度的标签来自固定集合：

- `correct-and-aligned`
- `reasonable-but-incomplete`
- `nothing-extracted`
- `contradiction-found`

只要任一维度出现 `contradiction-found`，当前步标签就会被视作错误步。

### 3.4 Hybrid Step Type Classifier：解释当前步骤属于哪一类

当前系统中的 `step type classifier` 仍然存在，它不是最终路由器，而是解释性分类层。

它的作用是：

- 判断当前步骤大概属于哪一类推理动作
- 为后续 specialist review 提供上下文
- 在 learned router 不可用时充当 fallback 路由来源

当前支持的步骤类型包括：

- `decomposition`
- `substitution`
- `algebraic_transformation`
- `condition_case`
- `final_conclusion`
- `arithmetic`
- `reasoning_transition`

当前模式支持：

- `heuristic`
- `llm`
- `hybrid`

其中 `hybrid` 的意思是：

- 先用规则版分类
- 对结构性或更复杂步骤尝试调用 LLM 分类
- 如果 LLM 分类失败，就回退到规则结果

### 3.5 Learned Router：直接学习 specialist 路由决策

这是在 `step type classifier` 之上新增的一层。

它与 step type classifier 的职责不同：

- `step type classifier` 负责“解释这一步像什么”
- `learned router` 负责“决定这一步该调哪些 specialist”

当前 learned router 的输入是一个 `RouterContext`，包括：

- `dataset`
- `question`
- `previous_steps`
- `current_step`
- `heuristic_step_type`
- `heuristic_risk_flags`
- `heuristic_specialists`

当前 learned router 的输出是一个 `RouteDecision`，包括：

- `selected_specialists`
- `trigger_specialist_review`
- `confidence`
- `source`
- `model_name`
- `candidate_scores`
- `fallback_used`
- `fallback_reason`

当前 specialist 候选只有三个：

- `alternative_route_verifier_tool`
- `equivalence_substitution_verifier_tool`
- `condition_obligation_verifier_tool`

另外 learned router 还会预测是否应该触发：

- `stage2_specialist_review`

### 3.6 Router 模式

当前 router 支持三种模式：

- `step-type`
- `learned`
- `learned-hybrid`

含义如下：

`step-type`

- 完全使用 step type classifier 派生出的 specialist 路由
- 不依赖 learned router 模型

`learned`

- 直接使用 learned router 输出 specialist 组合
- 不做低置信度 fallback

`learned-hybrid`

- 先尝试使用 learned router
- 如果 learned router 未配置、加载失败或置信度低于阈值
- 自动回退到 step type classifier 路由

这是当前最推荐的线上模式，因为它最稳。

### 3.7 Specialist Tools：局部结构化证据

当前 specialist 工具包括：

#### `alternative_route_verifier_tool`

用于判断：

- 当前步骤是否只是和参考路径不同
- 但依然是数学上可接受的替代路线

它更擅长处理：

- 多种正确解法
- 不同中间变量
- 不同拆分粒度

#### `equivalence_substitution_verifier_tool`

用于判断：

- 当前步骤中的代换、等价变形、表达式改写是否成立
- 是否存在硬数学冲突

它更擅长处理：

- 非等价变形
- 错误代换
- 假等式
- 不合法的结构改写

#### `condition_obligation_verifier_tool`

用于判断：

- 条件、分支、假设、结论义务是否满足

它更擅长处理：

- 条件遗漏
- 分情况不完整
- 结论跳步
- 不满足回答义务

### 3.8 Stage 2 Review：通用复核层

`stage2_review` 会基于：

- 原始 `stage2` 标签
- 当前步骤文本
- 局部 tool evidence

对原始 `stage2` 重新做一次复核。

它的目标是先修掉一部分明显由文本判断带来的粗误差。

### 3.9 Stage 2 Specialist Review：带路由与 evidence 的复核层

这一步是当前系统里非常关键的一层。

它的输入包括：

- 原始 `stage2` 标签
- `step type classifier` 输出
- `learned router` 输出
- specialist tool 的结构化 evidence

它的目标不是简单“看工具结果改标签”，而是：

- 站在 verifier agent 的角度重新判断这一整步
- 在“替代正确路径”和“真实硬冲突”之间做更稳的区分

### 3.10 Deterministic Specialist Adjustment：最后兜底

在 review 之后，系统还保留一层 deterministic fallback 纠偏。

当前主要规则是：

- 如果 specialist evidence 说明是 valid alternative，且没有硬冲突
  - 把 `contradiction-found` 降级成 `reasonable-but-incomplete`
- 如果 specialist evidence 说明存在 hard contradiction，而文本标签没抓到
  - 把对应维度升级成 `contradiction-found`

这层的作用是：

- 即使 review 层判断不稳定，也保留一个最小可解释的保底规则

### 3.11 每一步的持久化结果

当前每一步都会持久化下列重要字段：

- `stage2_step_type`
- `stage2_step_type_meta`
- `stage2_route_meta`
- `stage2_review_*`
- `stage2_specialist_review_*`
- `stage2_tool_trace`
- `parse_status`

其中 `stage2_route_meta` 很重要，里面保存了：

- 路由来源
- specialist 选择结果
- 是否触发 specialist review
- confidence
- fallback 情况
- candidate scores

这些字段是后续训练 learned router 的基础。

## 4. Learned Router 的训练目标

当前 learned router 不做生成式输出，而是做多标签分类。

训练目标共有 4 个标签：

- `use_alternative_route_verifier_tool`
- `use_equivalence_substitution_verifier_tool`
- `use_condition_obligation_verifier_tool`
- `trigger_specialist_review`

这意味着 learned router 学的不是“解释文字”，而是：

- 当前步骤到底该不该调某个 specialist
- 当前步骤到底该不该进入 specialist review

## 5. Router 训练数据是怎么构造的

### 5.1 Export 脚本

训练集导出脚本是：

```bash
python3 scripts/export_router_dataset.py
```

它会从已有 prediction jsonl 中读取每个 step 的信息，然后导出为 router 训练样本。

### 5.2 导出的标签策略

当前支持四种标签策略：

- `imitation`
- `weak-supervision`
- `benefit-aware`
- `expected-gain`

#### `imitation`

直接模仿当前系统已经实际走过的路由：

- 实际调了哪些 specialist
- 实际有没有触发 specialist review

这种方法的优点是简单、稳定。

缺点是：

- 它学习的是“历史系统怎么做”
- 不是“什么路由最有效”

#### `weak-supervision`

这是早于 `benefit-aware / expected-gain` 的收益代理版本。

它不是只模仿路由，而是综合以下收益信号来打标签：

1. specialist tool 的结构化 evidence
2. `stage2_original_parse -> final stage2_parse` 是否发生纠偏
3. `stage2_specialist_review` 是否真的改变了步级标签
4. 当 `gold_first_mistake_index` 可用时，是否让局部步级标签更接近 gold

举例来说：

- 如果 `alternative_route_verifier_tool` 给出了 `valid_alternative=true`
  - 会给 `use_alternative_route_verifier_tool` 加正分
- 如果 `equivalence_substitution_verifier_tool` 检测到硬冲突
  - 会给 `use_equivalence_substitution_verifier_tool` 加正分
- 如果 specialist review 让原本错误的步级标签变正确
  - 会给 `trigger_specialist_review` 更高权重
- 如果最终标签更接近 gold
  - 对参与过的 specialist 再加额外正分

如果某一步的弱监督信号非常弱，当前实现会自动：

- 回退到 `imitation_labels`

#### `benefit-aware`

这是在 `weak-supervision` 基础上增加 soft target 的版本。

它会进一步导出：

- `policy_targets`

这些 target 不再只是 0/1，而是会综合：

- specialist evidence
- `stage2_original_parse -> final stage2_parse` 的纠偏收益
- specialist review / deterministic adjustment 是否真的起作用
- 局部 gold 对齐是否改善或恶化

因此它比 `weak-supervision` 更接近“这个 route action 值不值得做”。

#### `expected-gain`

这是当前最新的默认导出策略，也是更接近 counterfactual route learning 的版本。

除了保留：

- `imitation_labels`
- `weak_labels`
- `policy_targets`

它还会额外导出：

- `expected_gain_targets`
- `supervision.expected_gain_confidence`
- `supervision.expected_gain_breakdown`

这一步的核心区别是：

- 不只看“已经发生的 action 有没有收益”
- 还会估计“没发生的 action 是否存在 missed opportunity”

例如：

- 某个 specialist 实际被调用，而且帮助了纠偏
  - 它的 `expected_gain_target` 会升高
- 某个 specialist 没被调用
  - 但 heuristic / step type 明显表明它相关
  - 且当前步最终仍与 gold 不一致
  - 它也会被打上更高的 `expected_gain_target`

因此它更像：

- expected-gain learning
- counterfactual action labeling

### 5.3 Sample Weight

导出的每个训练样本还会包含：

- `sample_weight`

当前权重逻辑大致是：

- 默认 `1.0`
- 如果 specialist review / adjustment 确实起作用
  - 提升到 `1.5`
- 如果明确改善了与 gold 的对齐
  - 提升到 `2.0`
- 如果 `expected_gain` 的置信度也比较高
  - 还会继续上调

这样训练时会优先学习“更可能真的有收益”的样本。

## 6. Router 训练脚本

训练脚本是：

```bash
python3 scripts/train_learned_router.py
```

### 6.1 当前默认基座模型

当前默认使用：

```text
Qwen/Qwen3-0.6B
```

训练方式是：

- `AutoModelForSequenceClassification`
- 多标签分类
- `LoRA`

### 6.2 当前训练损失

训练脚本使用的是：

- `BCEWithLogitsLoss`
- 多标签独立二分类

并在当前实现中支持：

- `sample_weight`

也就是说，每个样本的多标签 loss 会先算出来，再乘上样本权重。

### 6.3 训练输出

训练完成后会输出：

- router checkpoint
- tokenizer
- `mathchecker_router_config.json`

如果开启：

```bash
--save-merged-model
```

还会额外保存一份 merge 后模型。

## 7. 推理阶段如何使用训练好的 Learned Router

### 7.1 运行方式

训练好 router 后，可以在推理时这样接入：

```bash
uv run mathchecker run \
  --dataset big-bench-mistake \
  --model <your-llm-model> \
  --stage2-tools triad \
  --stage2-step-type-classifier hybrid \
  --stage2-router learned-hybrid \
  --stage2-router-model artifacts/router/qwen3-router \
  --stage2-router-threshold 0.55
```

这里要注意：

- `--model` 是主推理模型
- `--stage2-router-model` 是 learned router 模型

它们不是同一个东西。

### 7.2 推理时的决策顺序

在 `learned-hybrid` 模式下，推理顺序是：

1. 先跑 `stage1`
2. 再跑 `stage2 step type classifier`
3. 构造 `RouterContext`
4. 调用 learned router
5. 如果 learned router 置信度足够高
   - 用 learned router 的 specialist 选择
6. 如果 learned router 置信度不足或出错
   - 回退到 step-type 派生路由
7. 跑 `stage2`
8. 跑 `stage2_review`
9. 跑 `stage2_specialist_review`
10. 跑 deterministic specialist adjustment
11. 输出当前步最终标签

## 8. 训练示例

### 8.1 先导出 router 训练集

```bash
python3 scripts/export_router_dataset.py \
  --dataset big-bench-mistake \
  --model gpt-4o-mini \
  --output-dir artifacts \
  --export-path artifacts/router/bigbench_router_train.jsonl \
  --label-strategy expected-gain
```

### 8.2 安装 router 训练依赖

```bash
uv sync --group router
```

### 8.3 训练 Qwen3-0.6B Router

```bash
python3 scripts/train_learned_router.py \
  --train-jsonl artifacts/router/bigbench_router_train.jsonl \
  --output-dir artifacts/router/qwen3-router \
  --base-model Qwen/Qwen3-0.6B \
  --num-epochs 3 \
  --train-batch-size 4 \
  --eval-batch-size 8 \
  --learning-rate 2e-4 \
  --router-threshold 0.55
```

### 8.4 如果想禁用样本加权

```bash
python3 scripts/train_learned_router.py \
  --train-jsonl artifacts/router/bigbench_router_train.jsonl \
  --output-dir artifacts/router/qwen3-router-no-weight \
  --disable-sample-weights
```

## 9. 推理示例

### 9.1 示例输入

题目：

```text
If x = 3, compute x + 2.
```

前序步骤：

```text
(none)
```

当前步骤：

```text
Substitute x = 3 into x + 2 and simplify.
```

### 9.2 Step Type Classifier 输出示意

```json
{
  "step_type": "substitution",
  "reasoning": "The current step plugs a known value into an expression.",
  "risk_flags": ["substitution_risk", "equivalence_risk"],
  "confidence": 0.92
}
```

### 9.3 Router 输入文本示意

```text
Question: If x = 3, compute x + 2.
Previous steps:
(none)
Current step:
Substitute x = 3 into x + 2 and simplify.
Heuristic step type: substitution
Heuristic risk flags: substitution_risk, equivalence_risk
Heuristic specialist route: alternative_route_verifier_tool, equivalence_substitution_verifier_tool
```

### 9.4 Learned Router 输出示意

```json
{
  "selected_specialists": [
    "alternative_route_verifier_tool",
    "equivalence_substitution_verifier_tool"
  ],
  "trigger_specialist_review": true,
  "confidence": 0.91,
  "source": "learned_router",
  "candidate_scores": {
    "use_alternative_route_verifier_tool": 0.93,
    "use_equivalence_substitution_verifier_tool": 0.88,
    "use_condition_obligation_verifier_tool": 0.07,
    "trigger_specialist_review": 0.89
  }
}
```

### 9.5 Specialist Evidence 输出示意

假设此时 specialist 工具给出：

```json
[
  {
    "tool_name": "alternative_route_verifier_tool",
    "valid_alternative": true,
    "hard_contradiction": false
  },
  {
    "tool_name": "equivalence_substitution_verifier_tool",
    "valid_equivalent_transformation": true,
    "hard_contradiction": false
  }
]
```

这意味着：

- 当前步骤虽然可能和参考路径写法不同
- 但它更像是一个成立的替代代换步骤
- 不应该轻易打成 `contradiction-found`

### 9.6 最终效果示意

假设原始 `stage2` 判成：

```text
contradiction-found
```

但：

- specialist evidence 支持它是 valid alternative
- `stage2_specialist_review` 也重新判断它只是“不同但合理”

那么最终这一步可能被修正为：

```text
reasonable-but-incomplete
```

也就是说：

- learned router 决定了更合适的 specialist 组合
- specialist evidence 帮助识别“替代正确路径”
- review 层和最后的 deterministic adjustment 一起降低误判

## 10. 当前版本的优点

当前版本相较于最初的纯 prompt 判别，已经具备以下优势：

- 不再只依赖单次文本标签判断
- 可以利用 specialist tool 的结构化证据
- 对多种正确解法更友好
- 支持 step type classifier 与 learned router 分层协作
- 支持 learned-hybrid fallback，工程上更稳
- 已经具备从历史预测产物中自动导出训练集的能力
- 已经支持 `Qwen3-0.6B` 的第一版训练与加载
- 已经支持 `benefit-aware` router 训练目标
- 已经支持 `expected-gain` / counterfactual router 训练目标
- 已经具备独立的 `router evaluate` / `route ablation` 工具
- 已经支持 per-label threshold calibration 与回写模型配置

## 11. 当前版本的局限

当前版本仍有一些明显限制：

- learned router 仍然是第一代结构，主干仍是 `Qwen3-0.6B + multi-label sequence classification`
- 当前 `expected-gain` 已经开始编码 counterfactual missed opportunity，但仍然是离线代理收益，不是真正 interventional 的最优 route 标注
- 离线 `router evaluate` 已经补齐，但还没有完全和最终 `first mistake` 指标联合优化
- 不同数据集还没有单独 router head / dataset-specific router policy
- 还没有把 router 从“单步路由器”进一步升级成能做多轮查询的 verifier policy agent

## 11.1 新增升级

针对上一版的几个核心短板，当前实现已经新增以下能力：

- `expected-gain` 导出策略成为 router 数据集的默认导出方式
- 每条样本除了 `imitation_labels` / `weak_labels` 之外，还会导出：
  - `policy_targets`
  - `expected_gain_targets`
  - `supervision.policy_confidence`
  - `supervision.expected_gain_confidence`
  - `supervision.expected_gain_breakdown`
  - `supervision.benefit_signal_strength`
- 训练脚本支持直接选择训练目标字段：
  - `labels`
  - `imitation_labels`
  - `weak_labels`
  - `policy_targets`
  - `expected_gain_targets`
- router 输入不再只看题目和步骤文本，还会额外编码 `stage1` 的软指导：
  - `stage1_mathematical_concepts`
  - `stage1_key_analyses`
  - `stage1_calculations`
- 新增独立评测脚本：
  - `scripts/evaluate_router.py`
- 它可以离线比较：
  - `heuristic`
  - `imitation`
  - `oracle`
  - `learned`
  - `learned-hybrid`
- 同时还支持：
  - route-level precision / recall / F1
  - `avg_policy_utility`
  - `fallback_rate`
  - `expected calibration error`
  - `route ablation`
  - per-label threshold search
- `expected-gain` 标注不只提升已执行 action，也会显式编码未执行 action 的 missed opportunity
- learned router 现在还能从 `mathchecker_router_config.json` 读取 `per_label_thresholds`，使离线校准结果可以直接反馈回推理链路。

## 12. 一句话总结

当前系统的核心思想可以概括为：

> 先用 `stage1/stage2` 做基础步级判断，再用 `hybrid step type classifier` 提供解释性步骤分类，用带 `expected-gain` counterfactual 训练目标的 `learned router` 动态选择 specialist verifier，并让 specialist evidence 进入独立 review 层，最后通过 deterministic fallback 保留最小可解释兜底。

而 learned router 的训练目标不是“生成解释”，而是：

> 直接学习在什么样的步骤上下，应该调用哪些 specialist，以及是否应该触发 specialist review。
