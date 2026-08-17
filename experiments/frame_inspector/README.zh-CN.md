# Frame Inspector 实验

`Frame Inspector`（帧检查器）是 `Capture Lab`（画面采集实验台）的简易桌面监控界面。它直接消费 `experiments.capture_backends` 的统一后端契约，用于人工观察真实帧、采集健康状态和当前进程开销。

## 当前职责

- 选择 `mss`、`PrintWindow`、`WGC` 或 `dxcam` 后端；
- 选择一个可见顶层窗口、延时锁定前台窗口，或为 `mss`/`dxcam` 指定显示器索引；
- 在独立采集线程中连续获取 `FramePacket`（帧数据包）；
- 显示原始帧的实时预览和基础元数据；
- 显示滚动采集 FPS、当前/平均/P95 采集延迟和后端健康计数；
- 显示 Frame Inspector 进程的 CPU 与 RSS 内存占用；
- 按需保存当前原始帧的 PNG 和同名 JSON 元数据。

本工具不负责页面识别、录像、自动后端切换、输入记录、GPU占用采样或正式 `Trace Core` 接入。

## 运行

从主项目根目录执行：

```powershell
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe -m experiments.frame_inspector
```

在 PyCharm 中可创建“Python”运行配置：解释器选择 `D:\Games\worldtrace_workspace\.venv\Scripts\python.exe`，运行方式选择“模块名称”并填写 `experiments.frame_inspector`，工作目录填写 `D:\Games\worldtrace_workspace\worldtrace`。

界面启动后：

1. 选择截图后端；
2. 点击“刷新窗口”并选择目标，或选择“延时锁定前台窗口”；
3. 设置采集上限、预览刷新率和预览缩放；
4. 点击“开始监控”；
5. 需要保留证据时点击“保存当前帧”；
6. 点击“停止”，等待当前截图调用返回并释放后端。

保存位置：

```text
D:\Games\worldtrace_workspace\runtime_data\frame_inspector
```

## 输入与输出

输入：

```text
backend_name
CaptureTarget
CaptureConfig
target_fps
preview_fps
preview_scale
```

输出：

```text
实时帧预览
FramePacket元数据
CaptureHealth健康计数
CaptureMetrics滚动指标
可选PNG和JSON快照
```

预览刷新率和缩放只控制 GUI 显示成本。它们不会更改、压缩或降采样后端产生的原始 `FramePacket`，保存按钮仍保存完整原始帧。

## 指标口径

| 指标 | 当前含义 |
| --- | --- |
| 实际采集 FPS | 工作线程观察到成功交付帧的滚动频率 |
| 实际预览 FPS | GUI 真正完成画面刷新的滚动频率 |
| 当前/平均/P95 延迟 | `FramePacket.capture_latency_ns` 的滚动统计 |
| 尝试/成功/无帧/超时/失败 | 后端 `CaptureHealth` 的原始计数 |
| CPU | 当前 Frame Inspector 进程的总 CPU 百分比 |
| RSS | 当前进程驻留内存大小 |

不同后端的 `capture_latency_kind` 可能不同。例如 WGC 当前记录回调缓冲复制耗时，而同步后端记录调用耗时，因此这些值不能直接当成端到端延迟横向排名。

GPU占用显示为“未采集”，因为当前尚未定义统一采样来源和统计口径。

## 线程边界

`CaptureSession`（采集会话）在一个专用工作线程中创建后端，并在同一线程完成：

```text
open
-> start_stream
-> next_frame
-> stop_stream
-> close
```

GUI线程只发送停止信号，并通过有界队列取得最新帧和状态。队列满时替换旧预览帧，避免界面渲染较慢时无限堆积像素内存。不会在失败后静默切换到另一个后端。

启动和初始化阶段也会检查停止请求；如果后端工厂或 `open` 返回时用户已经停止，程序不会继续启动帧流。

`PrintWindow`、`mss` 和 `dxcam` 的同步系统调用不能被 Python 超时强制中断。点击停止后，界面会等待当前调用返回；若关闭窗口时调用仍未结束，后台线程会在调用返回后自行清理，并且不会阻塞独立 Frame Inspector 进程退出。

## 后端差异

- `mss` 和 `dxcam` 捕获窗口时会转换为桌面区域，因此可能包含遮挡内容；
- `PrintWindow` 只接受窗口目标，部分 GPU 游戏可能返回黑帧、拒绝或阻塞；
- `WGC` 使用窗口原生捕获范围，客户区/整窗选项不适用；
- 固定窗口在启动前会重新核对 HWND 对应的进程，延时前台锁定会拒绝选择 Frame Inspector 自身；
- WGC 或 DXcam 在静止画面下暂时没有新帧时会显示 `WAITING`，这不计作后端失败；
- `mss` 和 `PrintWindow` 当前无法确认来源帧是否为新帧，`freshness` 会显示 `UNKNOWN`；
- 自动锁定前台窗口只是方便选取目标，不代表规避或降低游戏反作弊风险。

## 测试

```powershell
# Frame Inspector内部测试
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe -m unittest discover `
  -s experiments\frame_inspector\tests -t . -p "test_*.py" -v

# 不打开可见窗口的GUI构造测试
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe `
  -m experiments.frame_inspector --smoke-test

# 所有实验测试
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe -m unittest discover `
  -s experiments -t . -p "test_*.py" -v
```

自动测试使用假后端验证线程归属、停止、错误处理、队列上限、FPS限速和指标计算，不会启动真实游戏。真实窗口的遮挡、最小化、黑帧、尺寸变化和颜色正确性仍需人工冒烟验证。
