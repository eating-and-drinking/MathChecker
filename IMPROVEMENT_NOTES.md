# 改进点记录

本文档用于汇总这个项目后续的所有方法改进点。
当前已记录多个点，后续可以继续按相同格式追加新的改进项。

## 改进点 1：让 `stage2` 对多种正确解法更友好

### 背景

- 原有 `stage2` prompt 更容易把 `stage1` 输出当作唯一参考路径。
- 在数学题存在多种等价正确解法时，容易把“不同但正确”的步骤误判为不对齐，甚至误判为矛盾。
- 这会影响错误步骤识别的准确性，尤其会增加“第一处错误定位”的误伤率。

### 修改内容

- 把 `stage1` 提供的内容从“唯一正确参考”降级为“软参考”。
- 要求模型优先判断当前步骤是否与题目、前序步骤、已有条件在数学上兼容。
- 明确说明以下情况本身不能直接视为 `contradiction-found`：
  - 与参考路径不同
  - 使用不同中间变量
  - 使用不同等价分解
  - 合并或拆分子步骤
  - 延后某些本可继续计算的表达式
- 收紧 `contradiction-found` 的使用范围，只在存在硬数学冲突时才允许使用，例如：
  - 算术错误
  - 非法代数变形
  - 符号或优先级错误
  - 与前序结论或题目条件冲突
  - 非等价替换或非等价分解
- 重新解释四个标签的含义，让 `correct-and-aligned` 能覆盖“不同但正确”的替代解法。

### 预期收益

- 降低多解法样本上的误报率。
- 减少“不同写法、不同路径、不同变量命名”导致的错误负判。
- 让 `contradiction-found` 更集中在真正的数学冲突上。
- 提高“第一处错误定位”任务中的稳定性和可信度。

### 风险

- prompt 变得更宽松后，可能会放过一部分本来应该判错的步骤。
- 是否真的提升整体效果，需要结合评测指标一起确认，不能只看个别样例。

### 影响文件

- `src/pedcot/pipeline/templates/stage2.txt`

### 验证方式

可以在相同模型、相同数据集下，对修改前后分别运行：

```bash
uv run pedcot run --dataset big-bench-mistake --model <your-model>
uv run pedcot evaluate --dataset big-bench-mistake --model <your-model>
```

重点观察：

- `MF_Acc`
- `Avg_F1`
- `Cls_Acc`
- 多解法样本是否更少被误判为 `contradiction-found`

### 可继续扩展的关联点

- 在 `stage1` 中生成“允许的多种下一步方案”，而不是单一路径参考。
- 在 `stage2` 中加入更显式的“valid-but-alternative”式中间判断。
- 调整最终步级聚合逻辑，避免“任一维度出现 `contradiction-found` 就直接判错”。
- 把 review 阶段的硬证据规则进一步前移并结构化。

## 改进点 2：升级为 step type classifier + specialist verifier agent 风格路由 + 独立证据 review 层

### 背景

- 仅靠 `stage2` 的文本标签判断，容易出现两类问题：
  - 不同但正确的替代解法，被误判为 `contradiction-found`
  - 文本判断没有识别出硬数学冲突，但局部工具其实已经能给出更强证据
- 之前的版本虽然已经接入了一批 tool-based specialist verifier，但整体仍偏向“rule-based / local tool + 一步纠偏”：
  - 缺少对步骤类型的显式识别，导致不同类型步骤复用同一套 verifier 组合
  - specialist evidence 主要直接进入最终标签修正，缺少一层独立的审查与再判断
  - 无法把“更像 verifier agent 的二次审阅能力”显式加入 `stage2` 判别链路

### 修改内容

- 保留并继续使用三个 `stage2` specialist verifier tool：
  - `alternative_route_verifier_tool`
  - `equivalence_substitution_verifier_tool`
  - `condition_obligation_verifier_tool`
- 这三个 verifier 分别负责：
  - 判断当前步骤是否是与参考路径不同但数学上成立的替代解法
  - 判断改写、代换、分解、等价变形是否成立，或是否存在硬冲突
  - 判断步骤是否违反题目条件、分支条件或局部结论义务
- 新增 `hybrid step type classifier`，采用“规则版 + LLM 分类 + 规则回退”的组合策略，先判断当前步骤类型，再决定触发哪些 specialist 组合。当前已支持的类型包括：
  - `decomposition`
  - `substitution`
  - `algebraic_transformation`
  - `condition_case`
  - `final_conclusion`
  - `arithmetic`
  - `reasoning_transition`
- 让 `step type classifier` 输出进入 `stage2` tooling 路由，按步骤类型动态选择 specialist，而不是所有步骤都走同一套 verifier：
  - 代换 / 变形类步骤优先触发 `alternative_route_verifier_tool` + `equivalence_substitution_verifier_tool`
  - 条件 / 分支 / 结论类步骤优先触发 `alternative_route_verifier_tool` + `condition_obligation_verifier_tool`
  - 其他一般步骤至少保留对替代正确路径更友好的 verifier 兜底
- 在此基础上，进一步加入第一版 `learned router` 骨架，把“步骤类型识别”和“specialist 路由决策”拆成两层：
  - `step type classifier` 继续负责解释性分类与 specialist review 上下文
  - `learned router` 直接负责选择 specialist 组合，并决定是否触发 `stage2_specialist_review`
- 当前 router 支持三种模式：
  - `step-type`
  - `learned`
  - `learned-hybrid`
- 其中 `learned-hybrid` 会在 learned router 低置信度、未配置模型或模型加载失败时，自动回退到现有 step-type 路由，避免破坏主链路稳定性
- 在原有 `stage2 review` 之外，再新增一层独立的 `stage2_specialist_review`：
  - 输入包含原始 `stage2` 标签
  - 输入包含 step type classifier 的输出
  - 输入包含 specialist router 的输出
  - 输入包含 specialist verifier 的结构化证据
  - 该层以“verifier agent 风格”的方式重新审阅当前步骤，而不是直接做硬编码改标签
- 当前整体链路升级为：
  - `stage2`
  - `stage2_review`
  - `stage2_specialist_review`
  - deterministic specialist fallback adjustment
- 也就是说，specialist evidence 现在会先进入独立 review 层，再进入最终聚合，而不再只是“一步纠偏”。
- 同时保留基于 tool trace 的 deterministic specialist adjustment 作为最后兜底：
  - 如果 specialist evidence 支持“valid alternative”且没有硬冲突，则把文本中的 `contradiction-found` 降级为 `reasonable-but-incomplete`
  - 如果 specialist evidence 给出硬数学冲突，而文本标签没有给出 `contradiction-found`，则自动把对应维度升级为 `contradiction-found`
- 新增 `stage2_route_meta` 持久化字段，把每一步的路由决策过程保存下来，包含：
  - route source
  - confidence
  - selected specialists
  - whether specialist review should trigger
  - fallback used / fallback reason
  - candidate scores
- 在 tool finalization instruction 中补充了 specialist tool 的使用规则，让模型更明确区分：
  - 硬冲突证据
  - 替代有效路径证据
  - 仅供参考的 advisory evidence
- 为了避免依赖额外符号库才有收益，还补了纯本地数值关系检查兜底，用于识别简单的假等式或假关系。
- 为 learned router 增加了第一版训练链路，当前采用 `Qwen/Qwen3-0.6B` 作为小型开源基座，路线为：
  - `sequence classification`
  - `multi-label routing`
  - `LoRA` 微调
- 训练目标当前不是自由生成 JSON，而是直接预测：
  - `use_alternative_route_verifier_tool`
  - `use_equivalence_substitution_verifier_tool`
  - `use_condition_obligation_verifier_tool`
  - `trigger_specialist_review`
- 为了支持训练，新增了两步式数据管线：
  - `scripts/export_router_dataset.py`
  - `scripts/train_learned_router.py`
- 数据导出目前支持两种标签策略：
  - `imitation`
  - `weak-supervision`
- 其中 `weak-supervision` 不再只模仿当前路由，而会综合以下信号构造训练标签：
  - specialist tool 的结构化 evidence
  - `stage2_original_parse -> final stage2_parse` 的纠偏变化
  - `stage2_specialist_review` 是否真的改变了步级判断
  - 在可判断时，是否让该步更接近 `gold_first_mistake_index` 对应的局部标签
- 同时为每条 router 训练样本生成：
  - `imitation_labels`
  - `weak_labels`
  - `sample_weight`
  - `supervision.reasons`
  - `supervision.label_scores`
- 在训练脚本中，`sample_weight` 会直接进入加权多标签损失，优先学习更可信的弱监督样本，而不是把所有样本一视同仁。

### 预期收益

- 降低“不同但正确”的步骤被误判为错误的概率。
- 提高对局部等价变形、替换、分解、条件义务错误的识别能力。
- 让不同类型步骤使用更匹配的 verifier 组合，而不是统一粗放处理。
- 让 specialist evidence 先经过独立审阅层整合，再进入最终标签输出，减少单步硬规则误伤。
- 通过保留 deterministic fallback，让系统在 review 层判断不稳定时仍有最小可解释兜底。
- 让 specialist 路由从“基于步骤类型的固定映射”逐步升级为“可训练的 learned router”，提高不同步骤上的动态适配能力。
- 为后续离线分析、路由消融和训练迭代提供更完整的 route-level 日志与可回放信号。
- 让 router 训练开始利用“是否真的帮助纠偏”这一类收益信号，而不是只学习历史系统的表面行为。
- 提高在复杂数学轨迹中定位第一处错误的稳定性和可诊断性。

### 风险

- tool 数量与 review 层级增加后，单步推理成本会更高。
- 当前 `hybrid step type classifier` 虽然已经引入 LLM，但规则层与提示词层仍需要继续校准，分类错误仍会影响 specialist 路由效果。
- 当前 learned router 还是第一版骨架，默认更像“可插拔的多标签路由器”而不是已经验证收益的最终模型。
- 当前 `weak-supervision` 标签仍然带有启发式成分，可能把系统现有偏差继续带进 learned router。
- 如果 specialist review prompt 设计不稳，可能把结构化证据重新“语言化”后带来新的波动。
- deterministic fallback 如果过强，仍可能覆盖掉更细腻的 review 结果，因此需要继续校准。
- 如果 learned router 的置信度阈值或回退策略设置不当，可能会出现：
  - 路由过于保守，学不到收益
  - 路由过于激进，覆盖掉原本稳定的 heuristic 路由
- 需要继续结合真实数据集做消融，确认 classifier、specialist review、fallback adjustment 各自的收益是否稳定。

### 影响文件

- `src/pedcot/pipeline/step_classifier.py`
- `src/pedcot/pipeline/router.py`
- `src/pedcot/pipeline/router_dataset.py`
- `src/pedcot/pipeline/tools.py`
- `src/pedcot/pipeline/predictor.py`
- `src/pedcot/pipeline/prompts.py`
- `src/pedcot/pipeline/__init__.py`
- `src/pedcot/core/models.py`
- `src/pedcot/core/constants.py`
- `src/pedcot/llm/openai_client.py`
- `scripts/export_router_dataset.py`
- `scripts/train_learned_router.py`
- `pyproject.toml`
- `tests/test_specialist_tools.py`
- `tests/test_step_classifier_and_review.py`

### 验证方式

可以在本地先确认基础行为：

```bash
uv run pytest
uv run pedcot --help
python3 scripts/export_router_dataset.py --help
python3 scripts/train_learned_router.py --help
```

再在真实数据集上比较改动前后：

```bash
uv run pedcot run --dataset big-bench-mistake --model <your-model> --stage2-tools triad
uv run pedcot evaluate --dataset big-bench-mistake --model <your-model>
python3 scripts/export_router_dataset.py --dataset big-bench-mistake --model <your-model> --export-path <router-train.jsonl>
uv sync --group router
python3 scripts/train_learned_router.py --train-jsonl <router-train.jsonl> --output-dir <router-checkpoint-dir>
uv run pedcot run --dataset big-bench-mistake --model <your-model> --stage2-tools triad --stage2-router learned-hybrid --stage2-router-model <router-checkpoint-dir>
```

重点观察：

- `MF_Acc`
- `Avg_F1`
- `Cls_Acc`
- `contradiction-found` 是否更少误伤替代解法
- tool evidence 是否能纠正原本漏掉的硬冲突
- step type classifier 是否把不同步骤送到更合理的 specialist 组合
- specialist review 是否比单纯 deterministic adjustment 更稳定
- learned router 是否能在高置信度步骤上优于 step-type 固定路由
- weak-supervision 导出的样本中，`sample_weight` 与实际纠偏收益是否大致一致
- learned-hybrid 模式下，fallback 触发频率是否合理

### 可继续扩展的关联点

- 继续把当前的 `benefit-aware` 软目标从离线代理收益，升级成更强的 counterfactual / expected-gain 标注。
- 把 `router evaluate` 的离线指标进一步和最终 `first mistake` 任务指标打通，形成更强的 end-to-end 选择准则。
- 把 specialist verifier 从“local tool + review prompt”继续升级为多轮 verifier agent。
- 为不同数据集配置不同的 specialist 组合、review 策略与聚合阈值。
- 让 specialist review 输出更结构化的 conflict span / evidence span，便于后续评测分析。

## 改进点 3：把 learned router 升级为 benefit-aware policy learner，并补齐 router evaluate / route ablation 闭环

### 背景

- 之前的 learned router 虽然已经能训练和推理，但还存在几个明显短板：
  - `labels` 仍然主要来自 imitation / heuristic weak supervision
  - 训练目标更像“学历史系统做过什么”，而不是“学什么 route 更有收益”
  - 还缺少独立的 router evaluate / route ablation 工具
  - 推理侧仍主要依赖单一全局阈值，缺少 route-level 校准
- 这会带来两个后果：
  - learned router 学到的是表面行为，不一定是收益最优策略
  - 没有单独评测闭环时，很难知道 router 改动到底提升了 route 质量，还是只是在拟合旧链路

### 修改内容

- 在 router 数据导出阶段引入 `benefit-aware` 标注策略，并将其升级为默认导出模式：
  - `scripts/export_router_dataset.py --label-strategy benefit-aware`
- 继续保留：
  - `imitation_labels`
  - `weak_labels`
- 同时新增：
  - `policy_targets`
  - `supervision.policy_confidence`
  - `supervision.benefit_signal_strength`
  - `supervision.benefit_learning_version`
- `policy_targets` 不再是硬二值标签，而是更接近 route policy 学习的 soft target：
  - 综合 specialist evidence
  - 综合 `stage2_original_parse -> final stage2_parse` 的纠偏收益
  - 综合 `specialist_review` / deterministic adjustment 是否真的起作用
  - 综合局部 gold 对齐是否改善或恶化
  - 对“被调用了但没有明显收益”的 action 给出更保守目标，而不是继续盲目抬高
- 为 learned router 增强输入上下文，不再只看：
  - `question`
  - `previous_steps`
  - `current_step`
  - heuristic step type / risk flags
- 现在额外把 `stage1` 的软指导也送进 router 输入：
  - `stage1_mathematical_concepts`
  - `stage1_key_analyses`
  - `stage1_calculations`
- 训练脚本 `scripts/train_learned_router.py` 现在支持：
  - `--target-field {labels, imitation_labels, weak_labels, policy_targets}`
  - route-level `compute_metrics`
  - 训练后自动落盘 `router_eval_metrics.json`
- 这意味着 router 训练不再只能看 `eval_loss`，还可以直接看：
  - `micro_precision`
  - `micro_recall`
  - `micro_f1`
  - `exact_match`
  - `avg_policy_utility`
  - `avg_selected_target`
- 新增离线 router 专用评测脚本：
  - `scripts/evaluate_router.py`
- 新脚本支持：
  - `heuristic`
  - `imitation`
  - `oracle`
  - `learned`
  - `learned-hybrid`
- 同时支持：
  - 独立 route-level 评测
  - `route ablation`
  - confidence / coverage 统计
  - expected calibration error
  - per-label threshold search
- 对 learned router 推理实现做了两项增强：
  - 新增 `score()` 接口，显式暴露 raw label score，便于离线评测和校准
  - 支持从 `pedcot_router_config.json` 读取 `per_label_thresholds`
- 这意味着现在可以先离线做 threshold calibration，再把结果回写到模型配置中，用于真实推理。

### 预期收益

- 让 learned router 从“模仿历史行为”更接近“学习哪些 route 更值得触发”。
- 让训练标签能表达不确定性和收益强弱，而不是只有 0/1。
- 让 router 训练与离线评测形成闭环，减少只能凭直觉改阈值和路由规则的情况。
- 通过 per-label threshold calibration，降低：
  - 某个 specialist 过度触发
  - 某个 specialist 长期触发不足
  - `trigger_specialist_review` 全局阈值不合适导致的过审或漏审
- 让 learned-hybrid 模式更容易逐步替代固定 heuristic 路由，而不是一次性硬切换。

### 风险

- `policy_targets` 仍然是收益代理信号，不是真正的 counterfactual 最优动作标签。
- 如果 soft target 的构造公式不稳，也可能把旧系统偏差以更隐蔽的方式带进训练。
- 离线 `avg_policy_utility`、`micro_f1` 提升，不一定自动等价于最终 `first mistake` 指标提升。
- per-label threshold calibration 如果只在单一数据切分上做，可能出现过拟合。

### 影响文件

- `src/pedcot/pipeline/router.py`
- `src/pedcot/pipeline/router_dataset.py`
- `src/pedcot/pipeline/router_eval.py`
- `src/pedcot/pipeline/predictor.py`
- `scripts/export_router_dataset.py`
- `scripts/train_learned_router.py`
- `scripts/evaluate_router.py`
- `tests/test_router_eval.py`

### 验证方式

先确认基础行为：

```bash
uv run pytest
python3 scripts/export_router_dataset.py --help
python3 scripts/train_learned_router.py --help
python3 scripts/evaluate_router.py --help
```

然后可以走一条更完整的 learned router 训练闭环：

```bash
python3 scripts/export_router_dataset.py --dataset big-bench-mistake --model <your-model> --export-path artifacts/router/train.jsonl
uv sync --group router
python3 scripts/train_learned_router.py --train-jsonl artifacts/router/train.jsonl --output-dir artifacts/router/qwen3-router
python3 scripts/evaluate_router.py --data-jsonl artifacts/router/train.jsonl --mode learned-hybrid --router-model artifacts/router/qwen3-router --optimize-thresholds --write-router-config
```

重点观察：

- `policy_targets` 是否比原先 `weak_labels` 更能体现收益强弱
- `avg_policy_utility` 是否高于 heuristic / imitation baseline
- `fallback_rate` 是否合理
- 某个 specialist 的 ablation 是否会明显降低 utility 或 F1
- per-label threshold calibration 后，`expected_calibration_error` 是否下降

## 改进点 4：把 benefit-aware 软目标继续推进为 expected-gain counterfactual 标注

### 背景

- 上一版 `benefit-aware` 已经比 `weak-supervision` 更接近收益学习，但它仍然主要依赖“已发生的 action 有无收益”。
- 这会遗漏一类很关键的训练信号：
  - 某个 specialist 没有被调用
  - 但从 step type / heuristic risk / 最终错误残留来看，它本来很可能值得调用
- 换句话说，上一版更像 observational soft target，还不够像 expected-gain learning 里的 counterfactual labeling。

### 修改内容

- 在现有：
  - `imitation_labels`
  - `weak_labels`
  - `policy_targets`
- 之外，新增：
  - `expected_gain_targets`
  - `supervision.expected_gain_confidence`
  - `supervision.expected_gain_breakdown`
  - `supervision.expected_gain_learning_version`
- 新的 `expected_gain_targets` 会同时建模两类信号：
  - observed gain
  - counterfactual missed opportunity
- 对已经执行的 action，会继续看：
  - specialist evidence 是否为正
  - route 是否真的改变
  - gold 对齐是否改善或恶化
  - action 是否在低相关场景被白白调用
- 对没有执行的 action，会新增估计：
  - 当前步骤是否对该 specialist 高相关
  - heuristic route 是否原本就建议它
  - 最终标签是否仍未解决与 gold 的偏差
  - 是否出现了 persistent contradiction / persistent route failure
- 这意味着现在导出的标签不只是“执行过的动作值不值得保留”，还会近似回答：
  - “如果补上这个动作，预期收益会不会更高”
- 为了让强 counterfactual 信号真正盖过旧 prior，还把 soft target 的融合方式改成动态混合：
  - 弱 gain 信号时，更信 prior
  - 强 gain / missed-opportunity 信号时，更信 expected gain 分量
- 同时把脚本链路一起打通：
  - `scripts/export_router_dataset.py` 新增 `--label-strategy expected-gain`
  - 并把它设为默认导出策略
  - `scripts/train_learned_router.py` 新增 `--target-field expected_gain_targets`
  - `scripts/evaluate_router.py` 新增 `--target-field expected_gain_targets`

### 预期收益

- 让 router 不只学“历史上做了什么有效”，还学“历史上没做、但本来应该做什么”。
- 降低因为旧系统 route 覆盖不足而导致的 imitation bias。
- 提高 learned router 对：
  - 未调用 specialist 的漏检
  - persistent false positive / false negative
  - review trigger 漏开
  这几类问题的学习能力。
- 让 `expected_gain_targets` 更适合作为下一阶段 learned router 的主训练目标。

### 风险

- 这仍然不是真正 online A/B 或 intervention 得到的因果收益标签。
- missed-opportunity 目前仍然基于 heuristic relevance 和最终残差误差推断，可能带来新的假阳性。
- 如果 counterfactual 信号权重调得太强，可能会过度反驳原本稳定的 observational prior。

### 影响文件

- `src/pedcot/pipeline/router_dataset.py`
- `src/pedcot/pipeline/router_eval.py`
- `scripts/export_router_dataset.py`
- `scripts/train_learned_router.py`
- `scripts/evaluate_router.py`
- `tests/test_router_eval.py`
- `LEARNED_ROUTER_ALGORITHM.md`

### 验证方式

先确认接口和测试：

```bash
uv run pytest
python3 scripts/export_router_dataset.py --help
python3 scripts/train_learned_router.py --help
python3 scripts/evaluate_router.py --help
```

然后可以直接用新目标训练：

```bash
python3 scripts/export_router_dataset.py --dataset big-bench-mistake --model <your-model> --export-path artifacts/router/train.jsonl
python3 scripts/train_learned_router.py --train-jsonl artifacts/router/train.jsonl --output-dir artifacts/router/qwen3-router --target-field expected_gain_targets
python3 scripts/evaluate_router.py --data-jsonl artifacts/router/train.jsonl --mode learned-hybrid --router-model artifacts/router/qwen3-router --target-field expected_gain_targets --optimize-thresholds
```

重点观察：

- `expected_gain_targets` 是否会在 missed-opportunity 样本上明显高于 `policy_targets`
- `avg_policy_utility` 是否继续提升
- route ablation 后，关键 specialist 的 utility 降幅是否更清晰
- threshold search 后，高置信度 coverage 和 calibration 是否更平衡
