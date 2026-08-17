# 数值单位、误差容限与按需AI分析构想

> 文档性质：OCR数值、单位类型、精度、确定性计算、偏差升级和复杂属性延迟分析的初步构想
>
> 材料规划基础：[材料规划、资源预算与公式知识构想](材料规划、资源预算与公式知识构想.md)
>
> 物品属性基础：[物品知识、实例账本与时效规则构想](../item_system/物品知识、实例账本与时效规则构想.md)
>
> 动态计算规则：[证据驱动的规则生成、实验验证与运行编译构想](../rule_synthesis/证据驱动的规则生成、实验验证与运行编译构想.md)
>
> 图标和状态证据：[图标视觉标记与状态证据构想](../interface_discovery/图标视觉标记与状态证据构想.md)
>
> 当前架构共识：[当前架构共识](../../architecture/当前架构共识.md)

## 1. 核心边界

系统处理游戏数值时，应遵循：

```text
确定性优先
-> 先观察、解析、换算、计算和复核

异常时升级
-> 只有有效观测与预期明显不符时才调用AI

复杂内容延迟分析
-> 先保存原始属性，用户需要时再调用AI
```

AI不应从第一帧开始参与普通数量、重量、百分比和面板数值解析。系统自身应该能够完成基础观察、简单推算、误差比较和多帧复核。

## 2. 两种合法AI入口

### 2.1 偏差驱动入口

系统执行、计算或预测后，游戏实际观测与预期值明显不符，并且已经排除常见OCR、单位、版本和上下文错误时，可以调用AI分析原因。

```text
确定性计算
-> 游戏实际观测
-> 容差比较
-> 多帧复核
-> 确定性诊断
-> 仍无法解释
-> AI_REVIEW_REQUIRED
```

### 2.2 用户按需入口

对于圣遗物词条、复杂装备组合、伤害收益或配装评价，系统先保存数据。用户主动提问时，再检索相关记录并调用AI。

```text
复杂属性被观察
-> 原始保存和基础解析
-> DEFERRED
-> 用户主动提问
-> AI按需分析
```

除这两种入口外，普通数值流程默认不调用AI。

## 3. 单位是重要线索，但不是唯一类型依据

系统可以根据单位帮助判断数值语义，例如：

```text
个、次、枚
-> 通常是离散整数

kg、m、s
-> 通常允许小数

%
-> 百分比或比例

伤害、攻击、防御
-> 可能显示整数，但中间计算包含小数
```

但单位不能单独决定存储类型：

- `1 kg`和`1.25 kg`使用相同单位但显示精度不同；
- 游戏可能把百分比显示为整数，但内部使用更高精度；
- 伤害最终显示整数，公式中间值可能是小数；
- 没有单位的圣遗物词条仍可能是百分比；
- 同一单位在不同游戏中可能采用不同取整规则。

因此还需要数值语义、显示精度、来源和公式上下文。

## 4. 数值种类

```text
NumericValueKind
├── INTEGER
├── DECIMAL
├── PERCENT
├── RATIO
├── DURATION
├── RANGE
├── DISTRIBUTION
├── APPROXIMATE
└── UNKNOWN
```

### 4.1 `INTEGER`

适用于物品数量、货币、兑换次数和离散等级。通常要求完全一致。

### 4.2 `DECIMAL`

适用于重量、距离、速度、部分面板属性和精确倍率。

### 4.3 `PERCENT`与`RATIO`

百分比需要同时保存原始显示和归一化表示：

```text
display_value = 12.5
display_unit = PERCENT
normalized_value = 0.125
```

不能把`12.5%`误当成倍率`12.5`。

### 4.4 `RANGE`

适用于掉落数量、刷新时间或说明中明确给出的上下限。

### 4.5 `DISTRIBUTION`

适用于随机掉落、暴击和其他概率结果，不能用单次实际结果直接否定期望值。

### 4.6 `APPROXIMATE`

适用于视觉估计、模型预测和只知道大致范围的数值。必须携带不确定性或误差范围。

## 5. 小数不等于使用普通`float`

界面显示小数，只说明数值不是离散整数，不代表实现必须使用二进制浮点数。

例如：

```text
1.2 kg
```

建议保存为：

```text
Decimal("1.2")
```

而不是依赖可能产生表示误差的普通`float`。

推荐原则：

```text
物品数量
-> integer

货币和界面小数
-> Decimal或定点数

精确比例
-> Decimal或有理数

模型内部推理
-> 可以使用float，但输出必须携带误差和单位
```

## 6. 单位定义

```text
UnitDefinition
├── unit_id
├── symbol_variants
├── dimension
├── canonical_unit_id
├── conversion_rule
├── display_precision_defaults
├── valid_value_kinds
├── game_profile_scope
└── version
```

可能的`dimension`：

```text
COUNT
MASS
LENGTH
TIME
CURRENCY
PERCENTAGE
DAMAGE
ATTRIBUTE_POINT
GAME_SPECIFIC
DIMENSIONLESS
```

游戏专用单位不能强行套入现实物理单位。系统可以为不同游戏建立独立`UnitDefinition`。

## 7. 原始值、显示值和归一化值

一个数值至少可能有三种表达：

```text
RawValue
-> OCR或游戏说明中的原始字符串

DisplayedValue
-> 按游戏界面精度解析后的值

NormalizedValue
-> 换算到标准单位或标准比例后的值
```

例如：

```text
RawValue = "+12.5%"
DisplayedValue = Decimal("12.5") percent
NormalizedValue = Decimal("0.125") ratio
```

三者都应保留，避免以后无法追溯换算和解析错误。

## 8. `NumericObservation`

```text
NumericObservation
├── observation_id
├── raw_text
├── parsed_value
├── value_kind
├── unit_id
├── normalized_value
├── display_precision
├── uncertainty
├── frame_id或window_id
├── roi_ref
├── ocr_confidence
├── parser_id与版本
├── game_context
└── evidence_ids
```

无法可靠识别小数点、正负号或百分号时，应保存多个候选：

```text
12.5
125
-12.5
```

并返回`UNKNOWN`或请求更多帧，不能为了继续计算强行选择一个候选。

## 9. OCR数值解析流程

```text
选择稳定帧或短时间窗
-> 裁剪目标数值区域
-> OCR读取原始文本
-> 字符和单位候选解析
-> 多帧投票或序列稳定
-> 合理范围检查
-> 上下文单位检查
-> 生成NumericObservation
```

常见风险包括：

- 小数点漏识别；
- `%`被识别成其他符号；
- `1`、`7`和斜线混淆；
- 千位分隔符和小数分隔符因语言不同而变化；
- 数值动画尚未结束；
- 增益、减益颜色被误当成正负号；
- 字体描边导致字符重复。

## 10. `NumericExpectation`

系统计算得到的预期值必须保留公式和上下文：

```text
NumericExpectation
├── expectation_id
├── expected_value_or_range
├── value_kind
├── unit_id
├── formula_or_rule_ref
├── formula_version
├── input_fact_refs
├── rounding_policy_ref
├── tolerance_policy_ref
├── applicable_context
└── created_at
```

如果预期值来自AI尚未验证的公式，必须明确标记为候选，不能与正式规则产生的预期值混为一类。

## 11. 显示取整和内部值

游戏可能执行：

```text
内部精确值
-> 中间步骤取整
-> 最终显示取整
```

因此比较前应先把预期值转换成游戏显示预期：

```text
expected_internal_value
-> apply_rounding_policy
-> expected_display_value
-> compare(observed_display_value)
```

不能直接用高精度内部预测值与只显示一位小数的面板文本比较。

## 12. `RoundingPolicy`

```text
RoundingPolicy
├── rounding_policy_id
├── mode
├── decimal_places
├── application_stage
├── minimum_increment
├── game_profile_scope
├── source_refs
└── version
```

可能的`mode`：

```text
ROUND_HALF_UP
ROUND_HALF_EVEN
FLOOR
CEIL
TRUNCATE
GAME_SPECIFIC
UNKNOWN
```

取整阶段也需要记录，因为每一步取整和最终统一取整可能产生不同结果。

## 13. 容差不能全局统一

不同数值需要不同`TolerancePolicy`：

```text
TolerancePolicy
├── tolerance_policy_id
├── absolute_tolerance
├── relative_tolerance
├── display_resolution
├── expected_uncertainty
├── warning_threshold
├── ai_review_threshold
├── minimum_recheck_count
├── applicable_property_refs
└── version
```

### 13.1 离散数量

```text
物品数量、货币、兑换次数
-> absolute_tolerance = 0
```

它们通常不允许“差一点”。

### 13.2 显示小数

如果界面显示一位小数，显示量化步长是`0.1`。在已知四舍五入规则时，可以根据半个显示步长建立比较边界。

### 13.3 派生游戏属性

面板数值可以同时考虑：

- 显示取整；
- 输入属性的OCR误差；
- 公式参数版本；
- 游戏可能存在的隐藏中间精度。

### 13.4 伤害预测

伤害可以同时使用绝对误差和相对误差，且必须固定敌人、增益、暴击和游戏状态后才有可比性。

## 14. 数值比较结果

```text
NumericComparisonResult
├── comparison_id
├── expectation_ref
├── observation_refs
├── normalized_expected
├── normalized_observed
├── absolute_difference
├── relative_difference
├── tolerance_policy_ref
├── outcome
├── reason_code
└── evidence_ids
```

`outcome`建议为：

```text
EXACT_MATCH
WITHIN_TOLERANCE
RECHECK_REQUIRED
AI_REVIEW_REQUIRED
USER_DECISION_REQUIRED
UNKNOWN
```

## 15. 偏差不应立即调用AI

发现偏差后的第一层流程：

```text
检查截图是否新鲜
-> 检查目标窗口和账号
-> 重新读取多个稳定帧
-> 检查OCR字符、小数点和单位
-> 检查单位换算
-> 检查公式、参数和游戏版本
-> 检查角色、装备、增益和游戏状态
-> 按正确取整顺序重新计算
-> 再次比较
```

只有这些检查无法解释且偏差超过`ai_review_threshold`时，才进入AI审核。

## 16. `NumericDeviation`

```text
NumericDeviation
├── deviation_id
├── expectation_ref
├── observation_refs
├── absolute_difference
├── relative_difference
├── tolerance_policy_ref
├── deterministic_checks
├── unresolved_conditions
├── deviation_status
├── ai_review_eligibility
└── evidence_ids
```

`deviation_status`可以是：

```text
DETECTED
RECHECKING
EXPLAINED_BY_OCR
EXPLAINED_BY_UNIT
EXPLAINED_BY_CONTEXT
EXPLAINED_BY_VERSION
UNEXPLAINED
AI_REVIEWED
USER_RESOLVED
```

## 17. AI升级门禁

```text
AIReviewTrigger
├── trigger_type
├── deviation_ref或user_query_ref
├── observation_health
├── repeated_sample_count
├── deterministic_check_summary
├── cost_budget
├── privacy_scope
└── allowed_context_refs
```

偏差驱动AI至少要求：

- 使用新鲜且有效的帧；
- 单位和数值候选已经收束；
- 重复观测仍然偏离；
- 当前上下文满足公式前置条件；
- 公式版本没有已知过期；
- 偏差超过该属性的AI阈值；
- 本次分析没有超出用户设置的AI预算。

## 18. 提供给AI的偏差包

AI不应只收到一句“为什么数值不对”。建议提供：

```text
NumericDeviationReviewBundle
├── expected_value
├── observed_values
├── raw_ocr_texts
├── units
├── formula_and_version
├── formula_inputs
├── rounding_policy
├── tolerance_policy
├── game_context
├── relevant_frames
├── deterministic_checks
├── known_counterexamples
└── user_question
```

AI可以提出：

- OCR仍可能存在误读；
- 单位或百分比换算错误；
- 公式遗漏增益、减益或隐藏系数；
- 取整发生在不同步骤；
- 游戏版本或角色状态发生改变；
- 当前样本不满足公式适用条件；
- 需要进行哪种额外观察或安全实验。

## 19. AI结果仍然是候选

```text
DeviationAnalysisArtifact
├── artifact_id
├── deviation_ref
├── proposed_explanations
├── proposed_formula_changes
├── requested_additional_evidence
├── model_id
├── source_refs
├── confidence
└── created_at
```

AI分析不能直接：

- 修改原始OCR观测；
- 覆盖正式公式；
- 改变库存数量；
- 调高容差以掩盖错误；
- 把无法解释的偏差强行标成正常；
- 在没有用户批准时执行新的高风险实验。

## 20. 用户确认

用户确认的是某个解释或修订候选：

```text
UserNumericDecision
├── decision_id
├── deviation_or_analysis_ref
├── selected_explanation
├── accepted_correction
├── create_new_rule_version
├── user_notes
└── confirmed_at
```

即使用户确认，也不覆盖历史数据。系统应创建：

- 新的数值观测修正记录；
- 新公式候选或正式版本；
- 新单位或取整规则版本；
- 对旧规则适用范围的限制。

## 21. 高级物品属性延迟分析

圣遗物、装备、随机词条和复杂套装效果可以先记录，不需要在发现时调用AI。

```text
DeferredAttributeBundle
├── bundle_id
├── item_type_id
├── item_instance_id或item_lot_id
├── raw_name
├── raw_description
├── basic_parsed_attributes
├── unresolved_attribute_lines
├── visual_token_refs
├── frame_and_roi_refs
├── game_version
├── parser_version
├── analysis_status
└── evidence_ids
```

`analysis_status`可以是：

```text
CAPTURED
BASIC_PARSED
DEFERRED
AI_ANALYZED
USER_CONFIRMED
STALE
```

## 22. 基础解析仍由系统完成

即使暂不调用AI，系统也可以完成：

- OCR读取物品名称和属性行；
- 识别整数、小数、正负号和百分号；
- 解析单位；
- 区分主词条和多个副词条的屏幕区域；
- 保存词条顺序；
- 对同一实例进行多帧去重；
- 记录无法识别的原始文本；
- 将简单数值绑定到对应属性候选。

无法确定“暴击率”和“暴击伤害”等语义时，可以保留机器属性ID和原文，不妨碍以后查询。

## 23. 用户按需查询

用户以后可以提出：

```text
这些圣遗物中哪些适合角色A
这个词条组合是否值得强化
两件装备哪一个预期伤害更高
当前库存能否组成某套配装
```

系统先由`AttributeQueryService`检索：

```text
相关ItemInstance和ItemLot
DeferredAttributeBundle
角色当前面板
已验证FormulaDefinition
BuildRecommendationClaim
材料和强化成本
游戏版本
```

只将与问题有关的数据提供给AI，避免扫描和解释全部库存。

## 24. `AnalysisArtifact`

```text
AnalysisArtifact
├── analysis_id
├── query
├── input_record_refs
├── model_id
├── formula_and_source_refs
├── assumptions
├── conclusions
├── uncertainties
├── recommendations
├── created_at
└── user_confirmation_status
```

AI分析结果与原始物品属性分开保存。用户未确认前，它只能用于解释和建议，不能成为永久基础属性。

## 25. 分析缓存和失效

相同问题可以复用已有`AnalysisArtifact`，但以下变化会使结果过期：

- 物品属性或强化等级改变；
- 角色面板改变；
- 配装和队伍改变；
- 游戏版本更新；
- 公式版本替换；
- 用户问题或目标改变；
- AI使用的攻略来源被标记为过期。

缓存键应包含相关记录和公式版本，不能只按自然语言问题缓存。

## 26. 简单系统推算范围

不调用AI时，系统应能够完成：

```text
单位解析与换算
加减乘除
批次取整
百分比和倍率换算
简单区间计算
多帧数值稳定
趋势和变化量
已验证公式求值
容差比较
库存前后对账
```

如果算法和输入明确，系统即使能够调用AI，也不应把普通算术交给AI。

## 27. 重量示例

```text
OCR观测：12.5 kg
规则预期：12.48 kg
界面显示精度：0.1 kg
取整方式：ROUND_HALF_UP

预期显示值：12.5 kg
结果：EXACT_MATCH
```

如果预期显示为`12.5 kg`，实际连续读取为`15.2 kg`，且单位、OCR和上下文均有效，则进入偏差流程。

## 28. 物品数量示例

```text
计划兑换后数量：35
实际稳定观测：34
value_kind = INTEGER
absolute_tolerance = 0

结果：RECHECK_REQUIRED
```

物品数量不应该因为“只差1个”就被容差接受。系统需要检查兑换批次、OCR、库存覆盖和是否发生其他消耗。

## 29. 人物面板示例

```text
公式预测攻击力：1234.46
游戏显示：1234
已确认显示规则：TRUNCATE到整数

预期显示：1234
结果：EXACT_MATCH
```

如果直接比较`1234.46`和`1234`，会产生没有意义的偏差。

## 30. 圣遗物示例

```text
物品名称：固定
主词条：攻击力 46.6%
副词条：暴击率 3.1%
副词条：暴击伤害 7.8%
副词条：元素精通 19
```

系统可以先准确保存名称、数值、百分号、词条区域和原始截图。是否适合某个角色、强化价值和伤害收益等问题，在用户提问时再调用AI与公式知识。

## 31. 随机结果不能按单值偏差处理

例如一次副本预期平均掉落`2.5`个材料，实际获得`2`个并不代表公式错误。

```text
expected_distribution
-> 与多次样本分布比较

single_observation
-> 只记录本次结果
```

只有积累足够样本后，才能判断掉落分布与预期是否存在系统性偏差。

## 32. 公式状态

```text
FormulaStatus
├── CANDIDATE
├── EXAMPLE_VERIFIED
├── GAME_TESTED
├── ACTIVE
├── DEGRADED
├── STALE
└── RETIRED
```

偏差发生时先确认当前预期是否来自`ACTIVE`公式。候选或过期公式产生的偏差不应被包装成游戏异常。

## 33. 与规则合成系统的连接

```text
NumericObservation
-> FactRecord

NumericExpectation
-> DerivedValueRule或FormulaDefinition求值

NumericDeviation
-> RuleHypothesis或公式修订候选

用户确认
-> 新RuleSpec或FormulaDefinition版本
```

数值表达式仍由白名单`ExpressionGraph`执行，AI只能提出候选表达式和解释。

## 34. 与物品系统的连接

```text
ItemTypeDefinition
-> 固定属性定义

ItemInstance
-> 随机和可变属性

DeferredAttributeBundle
-> 尚未进行语义分析的复杂属性证据

AnalysisArtifact
-> 用户按需产生的解释和建议
```

原始属性、基础解析和AI分析不得写入同一个字段。

## 35. 与材料规划的连接

材料计划中的数量、兑换价格和时间优先使用确定性类型：

```text
离散材料数量
-> INTEGER且零容差

生长和刷新时间
-> DURATION与明确ClockDomain

随机掉落
-> RANGE或DISTRIBUTION

商店倍率和折扣
-> DECIMAL或PERCENT
```

执行结果与计划偏差时，先进行库存和OCR对账；只有无法解释的大幅偏差才进入AI。

## 36. 与伤害公式的连接

伤害分析可以由AI帮助收集公式，但运行时仍然是：

```text
已确认面板观测
+ 已验证FormulaDefinition
+ 敌人和战斗上下文
-> 确定性预期伤害
-> 游戏实际观测
-> 容差和分布比较
-> 必要时AI分析
```

AI不能仅根据一张伤害数字截图重写伤害公式。

## 37. 故障和置信度隔离

至少分开：

```text
ocr_confidence
numeric_parse_confidence
unit_confidence
context_confidence
formula_confidence
tolerance_policy_confidence
observation_freshness
comparison_confidence
```

以下情况不能作为公式反例：

- 截图失败或旧帧；
- OCR小数点和单位不确定；
- 角色或装备上下文错误；
- 游戏处于临时增益状态；
- 公式版本已过期；
- 随机结果样本不足；
- 画面数值仍在动画变化中。

## 38. AI成本和触发频率

系统应避免对每次小偏差重复调用AI：

```text
同类偏差聚合
-> 保存证据
-> 达到重复次数或用户请求
-> 一次性生成ReviewBundle
```

对于已经存在且仍适用的`AnalysisArtifact`，可以先向用户展示缓存结论和版本，再决定是否重新调用AI。

## 39. 分阶段验证

### 阶段一：基础数值类型

验证整数、Decimal、百分比、时间和范围可以正确解析、换算和序列化。

### 阶段二：多帧OCR数值

选择一个稳定面板数值，验证小数点、百分号、单位和多帧收束。

### 阶段三：显示取整

选择一个已知内部计算和界面显示不同的属性，验证比较发生在正确显示精度。

### 阶段四：零容差数量

验证物品数量相差1时不会被普通小数容差接受。

### 阶段五：偏差确定性诊断

注入单位错误、OCR错误和版本错误，确认它们在调用AI前被发现。

### 阶段六：AI偏差审核

构造一个确定性流程无法解释的重复偏差，只向AI提供结构化ReviewBundle，检查输出是否保持候选性质。

### 阶段七：复杂属性延迟保存

记录一批圣遗物属性，不调用AI，验证原文、数值、图标和实例引用完整。

### 阶段八：用户按需分析

用户选择少量实例提出具体问题，系统只检索相关数据和公式调用AI，并保存独立`AnalysisArtifact`。

### 阶段九：分析失效

改变物品等级、角色面板或公式版本，验证旧分析被标记为`STALE`。

## 40. 第一版验收标准

- 单位参与类型判断，但不会成为唯一判断依据；
- 小数优先使用Decimal或定点表示；
- 原始OCR、显示值和归一化值全部可追溯；
- 物品数量和货币保持零容差；
- 面板比较会应用正确显示精度和取整顺序；
- 每种属性拥有独立容差策略；
- 单次偏差不会立即调用AI；
- OCR、单位、上下文和版本会先被确定性复核；
- 只有重复且明显的未解释偏差进入AI审核；
- AI分析不能覆盖原始观测和正式公式；
- 用户确认后创建新版本而不是修改历史；
- 复杂圣遗物属性可以长期保持`DEFERRED`；
- 用户主动提问时只调用相关数据；
- AI分析结果作为独立`AnalysisArtifact`保存；
- 普通观察、换算、算术和比较在没有AI时仍能运行。

## 41. 当前结论

数值系统应坚持：

```text
观察原文
-> 确定性解析
-> 单位和精度归一化
-> 已验证规则计算
-> 属性专用容差比较
-> 确定性复核
-> 必要时AI分析
-> 用户确认并生成新版本
```

系统自身负责简单、重复、可验证的工作。AI只处理反复复核后仍无法解释的数值偏差，以及用户主动要求理解的复杂属性组合。
