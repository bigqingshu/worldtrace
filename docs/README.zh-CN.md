# 界迹文档导航

本目录使用英文文件夹保证工具链、导入和跨平台路径稳定，MD 文件名与正文保留中文。阅读和检索时不要求先记住英文目录，可从本页进入。

## 推荐阅读顺序

1. [项目初始目标与阶段路线](architecture/项目初始目标与阶段路线.md)
2. [当前架构共识](architecture/当前架构共识.md)
3. [程序事实记忆、AI智能体记忆与写入治理](architecture/程序事实记忆、AI智能体记忆与写入治理.md)
4. [构想收束、可观测性审计与实验方法](architecture/构想收束、可观测性审计与实验方法.md)
5. [第一阶段开源采集验证与实现路线](architecture/第一阶段开源采集验证与实现路线.md)
6. [项目可行性、周期、风险与阶段成功标准](architecture/项目可行性、周期、风险与阶段成功标准.md)
7. [总体架构与关键问题分析](concepts/game_ai_planning/01-总体架构与关键问题分析.md)
8. [证据驱动的规则生成、实验验证与运行编译构想](concepts/rule_synthesis/证据驱动的规则生成、实验验证与运行编译构想.md)
9. [分层关系网与复合能力子图构想](concepts/interface_discovery/分层关系网与复合能力子图构想.md)
10. [游戏瞬态故障判读与置信度隔离构想](concepts/interface_discovery/游戏瞬态故障判读与置信度隔离构想.md)
11. [图标视觉标记与状态证据构想](concepts/interface_discovery/图标视觉标记与状态证据构想.md)
12. [物品知识、实例账本与时效规则构想](concepts/item_system/物品知识、实例账本与时效规则构想.md)
13. [材料规划、资源预算与公式知识构想](concepts/resource_planning/材料规划、资源预算与公式知识构想.md)
14. [数值单位、误差容限与按需AI分析构想](concepts/resource_planning/数值单位、误差容限与按需AI分析构想.md)
15. [基于相对深度的地形可通行性判断构想](concepts/depth_geometry/基于相对深度的地形可通行性判断构想.md)
16. [人物运动观测、跳跃响应学习与分析窗口构想](concepts/depth_geometry/人物运动观测、跳跃响应学习与分析窗口构想.md)
17. [离散移动状态、视觉推进证据与条件化置信度构想](concepts/depth_geometry/离散移动状态、视觉推进证据与条件化置信度构想.md)
18. [3D场景识别与视觉地点定位技术参考](references/visual_localization/3D场景识别与视觉地点定位技术参考.md)
19. [三类3D场景寻路构想综合分析与实施建议](concepts/navigation_3d/三类3D场景寻路构想综合分析与实施建议.md)
20. [外部视频攻略解析、场景对齐与闭环复现构想](concepts/guide_following/外部视频攻略解析、场景对齐与闭环复现构想.md)
21. [目录与模块边界规则](architecture/目录与模块边界规则.md)

## 文档分类

| 英文目录 | 中文含义 | 当前内容 |
| --- | --- | --- |
| `architecture` | 已确认架构 | 项目目标、当前共识、程序事实与AI记忆治理、收束方法、实施路线、风险周期和模块边界 |
| `concepts/foundations` | 原始构想 | 最初想法和早期目录讨论记录 |
| `concepts/game_ai_planning` | 游戏AI规划构想 | 总体架构、图像处理、调度、关系网和执行方案，共14篇 |
| `concepts/interface_discovery` | 界面与交互发现 | 控件发现、窗口输入、故障隔离、动态图标绑定、状态证据和视角控制，共8篇 |
| `concepts/depth_geometry` | 深度几何构想 | 相对深度、局部地形可通行性、人物运动、离散推进状态与跳跃响应学习，共3篇 |
| `concepts/item_system` | 物品系统构想 | 物品知识、实例账本、获取规则、时效和消失归因，共1篇 |
| `concepts/navigation_3d` | 3D场景导航 | 视觉移动、立体地图、无小地图副本和综合建议，共4篇 |
| `concepts/guide_following` | 外部攻略构想 | 外部攻略解析、场景对齐与闭环复现，共1篇 |
| `concepts/resource_planning` | 资源规划构想 | 材料计算、时间排程、数值容差、按需AI分析和后期公式知识，共2篇 |
| `concepts/rule_synthesis` | 规则合成构想 | 从事实和实验生成状态、关系、公式并确定性编译，共1篇 |
| `references/whimbox` | Whimbox参考 | 项目架构、源码专题和交互控制分析，共19篇 |
| `references/alas` | ALAS参考 | 可借鉴机制与设计思想 |
| `references/visual_localization` | 视觉定位参考 | 场景分类、视觉地点识别、几何核验、SfM与SLAM技术梳理，共1篇 |
| `decisions` | 设计决策 | 后续保存已正式采用的 ADR，目前为空 |
| `modules` | 模块说明 | 后续保存已实现模块的职责和接口，目前为空 |

## 迁移说明

本次从 Whimbox 工作区复制了相关分析文档，原文件全部保留。跨文档链接已经调整到当前英文目录。

Whimbox源码链接按以下预期位置组织：

```text
D:\Games\worldtrace_workspace\reference_repos\whimbox
```

在参考仓库尚未放入该位置前，文档内部的源码链接会暂时不可打开，但文档之间的链接不受影响。

## 文档维护规则

- 一个概念只使用一个固定英文名称，统一翻译见[术语表](术语表.md)。
- 构想尚未成为正式决定时，只能放在 `concepts`。
- 正式采用或否决的重要方案进入 `decisions`，并记录原因和影响。
- 模块实现后，在 `modules` 中建立对应中文说明，写明职责、输入、输出、依赖和不负责范围。
- AI提到英文目录、模块或类型时，首次出现应同时附带中文含义。
