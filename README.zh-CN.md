# 界迹（WorldTrace）

界迹是一个面向游戏的视觉关系网学习、规划与执行系统。当前项目处于架构收束和最小闭环准备阶段，首期目标是完成可记录、可对齐、可保存和可回放的 `Trace Core`。

## 建议先读

1. [项目初始目标与阶段路线](docs/architecture/项目初始目标与阶段路线.md)
2. [当前架构共识](docs/architecture/当前架构共识.md)
3. [程序事实记忆、AI智能体记忆与写入治理](docs/architecture/程序事实记忆、AI智能体记忆与写入治理.md)
4. [构想收束、可观测性审计与实验方法](docs/architecture/构想收束、可观测性审计与实验方法.md)
5. [第一阶段开源采集验证与实现路线](docs/architecture/第一阶段开源采集验证与实现路线.md)
6. [项目可行性、周期、风险与阶段成功标准](docs/architecture/项目可行性、周期、风险与阶段成功标准.md)
7. [目录与模块边界规则](docs/architecture/目录与模块边界规则.md)
8. [文档导航](docs/README.zh-CN.md)
9. [中英文术语表](docs/术语表.md)

## 工作区结构

| 英文目录 | 中文理解 | 主要内容 |
| --- | --- | --- |
| `worldtrace` | 主项目 | 自有源码、测试和可版本化文档 |
| `reference_repos` | 参考仓库 | 保持原样的 Whimbox、ALAS、UIAgent 等项目 |
| `reference_forks` | 参考分支 | 确实需要修改的外部项目副本 |
| `environments` | 运行环境 | Python 虚拟环境 |
| `model_store` | 模型仓库 | OCR、YOLO、CLIP、SAM 等模型权重 |
| `data_store` | 数据仓库 | 录屏、截图、标注集和回放数据 |
| `runtime_data` | 运行数据 | 日志、证据包、数据库和缓存 |
| `scratch` | 临时试验区 | 可丢弃的一次性验证内容 |

## 当前开发原则

- 目录和源码标识符使用英文，解释文档使用中文。
- 新想法先进入 `docs/concepts`，技术试验先进入 `experiments`。
- 构想进入实现前，先完成可观测性映射和最小实验设计。
- 只有输入、输出和验收方式明确后，试验代码才能迁入 `src/worldtrace`。
- `main.py` 只负责启动和依赖组装，不承载业务算法。
- 第一阶段不接入 OCR、YOLO、CLIP、SAM 或 LLM。
- 程序事实与AI记忆分离；AI只读取带revision的只读视图并提交候选，正式写入由程序服务校验和执行。
- 当前不预先建立未来模块空壳，按通过验证的能力逐步扩展。
