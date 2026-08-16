# Pointer Context Lab（指针上下文实验）

## 实验目标

`experiments.pointer_context_lab` 是独立的只读指针上下文检测实验。第一版只回答：

> 在用户显式选择一个普通顶层窗口后，能否周期读取 Windows 可公开观察的指针、
> 光标、裁剪区域和传统鼠标捕获信号，并将它们组合为非权威的输入上下文候选？

它不是单独判断“鼠标有没有被锁定”，而是区分当前画面更接近哪一种可观察的
`PointerContextCandidate`（指针上下文候选）：普通可定位 UI、锁定式相对指针、
两者混合／切换，或证据不足。它帮助后续人工选择
`POSITIONED_UI_CLICK`（定位式 UI 点击）或
`LOCKED_POINTER_CLICK`（锁定指针点击）实验语义。它不会自动改写输入方案，也
不会把候选写成确认的 2D／3D 游戏状态。

本实验保留在 `experiments`；没有 Trace Core 公共契约变化。

## 输入与输出

输入：

- 用户显式选择的可见顶层窗口；
- 选择时冻结的
  `HWND + PID + process_started_at + title + client_region`；
- Windows 公开接口返回的只读信号；
- 当前实验会话的 `focus_epoch`（焦点代次）；
- 多次连续采样形成的稳定持续时间窗口。

输出：

- `PointerContextCandidate`（指针上下文候选）；
- 稳定或未稳定的进程内判断；
- 最新一份 `PointerContextSnapshot`（指针上下文快照）；
- 冻结客户区与当前客户区的 `left/top/width/height` 差值；
- GUI 中覆盖显示的当前原始 JSON；
- 最后一份“同一 HWND、PID 和进程实例确实处于前台”时的原始 JSON。

GUI 不追加完整采样历史；会话内部只保留有界的易失快照用于测试和诊断，不写入
文件。“最后目标前台快照”只是带序号和时间差的易失副本，不会被冒充为当前状态。
停止后界面仅保留最后一次屏幕显示，关闭进程即消失。

## 第一版候选

| 候选 | 含义 |
| --- | --- |
| `POSITIONED_UI_CANDIDATE` | 当前证据更接近可见、可定位的普通 UI 指针 |
| `LOCKED_RELATIVE_CANDIDATE` | 光标隐藏，并同时观察到目标客户区裁剪或目标窗口捕获 |
| `HYBRID_OR_TRANSITION` | 证据混合，可能处于菜单、覆盖层或模式切换 |
| `UNKNOWN` | API 失败、目标关系异常或证据不足／冲突 |

`GetClipCursor` 成功返回的裁剪证据分为正常矩形和单点两种形状。部分全屏或游戏
窗口会观察到零宽、零高的原生 `RECT`；实验将其保存为 `clip.shape=POINT` 和
原始屏幕点，不再冒充 API 失败。只有当该点位于目标客户区、接近当前光标位置，
并同时观察到光标隐藏时，它才参与锁定式相对指针候选；该组合仍只是候选证据，
不是微软契约对零面积矩形的权威语义，也不证明游戏使用了 Raw Input。

候选必须跨连续采样保持到约定时长才可显示为稳定。失焦、目标身份变化、客户区变化、
API 失败或候选变化都不能沿用旧稳定结论。

高频轮询可能在 Windows 可见的同一单调时钟刻度内得到两个相同时间戳；会话只将
后一个样本顺延 1 ns 以维持严格顺序，不把时钟量化冒充为提供器失败。只有时间戳
真正倒退时才记录 `PROVIDER_ERROR` 并阻止沿用稳定结论。

客户区位置和尺寸采用不同策略：

- 只有 `left/top` 变化时，记录 `TARGET_POSITION_CHANGED` 和精确差值，使用当前
  客户区继续只读判断，并从该位置重新累计稳定时间；
- `width/height` 变化时，记录 `TARGET_GEOMETRY_CHANGED` 并保持 `UNKNOWN`；
- 同一采样可同时列出位置、尺寸、失焦等全部门禁原因，不再只返回最先命中的原因。

这里的位置重新基线只属于检测会话，不会更新输入方案、点击坐标或 Trace Core 状态。

## GUI 操作

1. 点击“刷新窗口”并显式选择目标；
2. 点击“开始只读轮询”；
3. 查看候选、稳定性、全部原因和几何差值；
4. 切换游戏菜单与玩法画面，观察候选是否经过暂态后发生变化；
5. 切回本 GUI 后，在“最后目标前台快照”标签读取切回前保留的有效样本；
6. 点击“停止轮询”后再选择其他窗口。

运行期间冻结目标身份，不按标题自动重连。一个采样失败会显示 `UNKNOWN` 并保留
继续轮询的机会，不会伪造上一次稳定候选。

## 安全与数据边界

第一版：

- 不发送键盘或鼠标输入；
- 不调用 `SendInput`；
- 不注册 Raw Input；
- 不安装输入钩子；
- 不注入或读取目标进程；
- 不截图、不录屏；
- 不把 JSON、日志或候选历史写入磁盘；
- 不自动申请管理员权限；
- 不判断、绕过或排除反作弊；
- 不声称目标游戏消费了任何输入。

光标隐藏、裁剪区域或传统 `hwndCapture` 单独出现都不足以确认 3D 镜头模式。
若游戏使用 Raw Input、DirectInput、自绘光标或其他引擎内部路径，第一版可能保持
`UNKNOWN`。本实验不直接检测 Raw Input，也不验证镜头是否移动或目标是否消费输入。
后续物理相对位移和视觉响应应分别进入新的实验，不能偷偷扩展本 GUI。

## 启动

```powershell
cd D:\Games\worldtrace_workspace\worldtrace
..\.venv\Scripts\python.exe -m experiments.pointer_context_lab
```

无原生采样的离屏 GUI 冒烟：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
..\.venv\Scripts\python.exe -m experiments.pointer_context_lab --smoke-test
```

## 自动化验收

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
..\.venv\Scripts\python.exe -m unittest discover `
  -s experiments\pointer_context_lab\tests `
  -p "test_*.py" `
  -t . `
  -v

ruff check experiments\pointer_context_lab
ruff format --check experiments\pointer_context_lab
```

自动测试使用注入的窗口、native provider（原生信号提供器）和 session
（会话）假对象，不调用真实 Windows 指针 API，也不发送输入。
