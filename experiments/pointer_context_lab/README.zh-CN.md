# Pointer Context Lab（指针上下文实验）

## 实验目标

`experiments.pointer_context_lab` 是独立的只读指针上下文检测实验。第一版只回答：

> 在用户显式选择一个普通顶层窗口后，能否周期读取 Windows 可公开观察的指针、
> 光标、裁剪区域和传统鼠标捕获信号，并将它们组合为非权威的输入上下文候选？

它用于区分“更像定位式 2D UI 指针”与“更像锁定式 3D 相对指针”的观察条件，
帮助后续人工选择 `POSITIONED_UI_CLICK`（定位式 UI 点击）或
`LOCKED_POINTER_CLICK`（锁定指针点击）实验语义。它不会自动改写输入方案，也
不会把候选写成确认的游戏状态。

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
- GUI 中覆盖显示的最新原始 JSON。

GUI 不追加完整采样历史；会话内部只保留有界的易失快照用于测试和诊断，不写入
文件。停止后界面仅保留最后一次屏幕显示，关闭进程即消失。

## 第一版候选

| 候选 | 含义 |
| --- | --- |
| `POSITIONED_UI_CANDIDATE` | 当前证据更接近可见、可定位的普通 UI 指针 |
| `LOCKED_RELATIVE_CANDIDATE` | 光标隐藏，并同时观察到目标客户区裁剪或目标窗口捕获 |
| `HYBRID_OR_TRANSITION` | 证据混合，可能处于菜单、覆盖层或模式切换 |
| `UNKNOWN` | API 失败、目标关系异常或证据不足／冲突 |

候选必须跨连续采样保持到约定时长才可显示为稳定。失焦、目标身份变化、客户区变化、
API 失败或候选变化都不能沿用旧稳定结论。

## GUI 操作

1. 点击“刷新窗口”并显式选择目标；
2. 点击“开始只读轮询”；
3. 查看候选、稳定性和最新原始 JSON；
4. 切换游戏菜单与玩法画面，观察候选是否经过暂态后发生变化；
5. 点击“停止轮询”后再选择其他窗口。

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
`UNKNOWN`。后续物理相对位移和视觉响应应分别进入新的实验，不能偷偷扩展本 GUI。

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
