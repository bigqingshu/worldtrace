# Capture Backends 实验

本目录实现 `Capture Lab`（画面采集实验台）的第一块最小能力：通过可替换后端获取一帧图像，并把像素和采集元数据一起输出。

## 当前范围

- `mss`：显示器、桌面区域，以及转换为桌面区域后的窗口客户区；
- `PrintWindow`：原生窗口句柄；
- `WGC`：通过 `windows-capture 2.x` 按窗口句柄采集；
- `dxcam`：显示器和输出内区域，以及单显示器条件下转换后的窗口区域；
- 统一生成实验版 `FramePacket`（帧数据包）；
- 保存 PNG 和同名 JSON 元数据。

本包本身不包含 GUI、性能计数器、录像、旧帧缓存、自动后端降级或正式 `Trace Core` 接入。上层实时预览和性能监控位于 [`experiments/frame_inspector`](../frame_inspector/README.zh-CN.md)，不会反向进入后端适配器。

## 目录职责

```text
contracts.py
-> 实验内部数据契约和错误码

target_selector.py
-> Windows窗口枚举、前台窗口选择和客户区坐标

backends/
-> 四种采集API的独立适配器

registry.py
-> 延迟加载和能力探测

image_writer.py
-> FramePacket到PNG和JSON

__main__.py
-> 单帧命令行入口
```

缺少任一可选后端依赖时，其他后端仍可使用。程序不会静默切换后端，因为那会改变截图来源和实验语义。

同一个后端实例的`open/start_stream/next_frame/stop_stream/close`必须全部在同一个采集线程调用。GUI阶段应向采集工作线程发送命令，不能在UI线程打开后端、再到工作线程取帧。

## 环境

当前工作区使用项目级独立环境：

```powershell
D:\Games\worldtrace_workspace\.venv\Scripts\python.exe -m pip install -r environment_specs\capture_lab-requirements.txt
```

不要把 WorldTrace 的依赖安装进 Whimbox、UIAgent 或其他项目环境。

## 使用

从主项目根目录运行：

```powershell
# 查看依赖与后端可用性
python -m experiments.capture_backends --list-backends

# 查看可选择的可见顶层窗口
python -m experiments.capture_backends --list-windows

# 倒计时3秒后读取一次前台窗口并锁定，然后截取客户区
python -m experiments.capture_backends --backend mss --foreground

# 使用十六进制窗口句柄
python -m experiments.capture_backends --backend printwindow --hwnd 0x123456

# 截取第一块显示输出
python -m experiments.capture_backends --backend dxcam --display 0

# WGC原生窗口采集
python -m experiments.capture_backends --backend wgc --hwnd 0x123456
```

未提供 `--output` 时，文件写入工作区的 `runtime_data/capture_lab`，不会写入源码目录。

## 已知边界

- `mss`和`dxcam`的窗口目标仍是桌面区域，会包含遮挡窗口；
- `PrintWindow`可能被GPU游戏拒绝、返回黑帧或同步阻塞；
- `--frame-wait-timeout`只限制无帧重试和事件等待，不能强制中断已经进入的同步`PrintWindow`调用；
- 当前`dxcam`窗口区域只支持设备0、输出0坐标，跨显示器会明确拒绝；
- WGC回调中的借用缓冲会立即复制为自有CPU字节，尚未保留GPU纹理；
- WGC窗口项按库的原生窗口范围交付；元数据保留请求范围，并把实际目标标为`NATIVE`，不猜测为客户区或整窗；
- `capture_latency_kind`区分同步API调用耗时与WGC回调复制耗时，二者不能直接比较；
- `mss`和`PrintWindow`没有来源帧时间戳，当前记录调用开始、完成时间和完成时的单调时钟；
- `mss`和`PrintWindow`返回帧的`freshness`为`UNKNOWN`，不会把调用成功冒充为已确认的新来源帧；
- 自动选择前台窗口只是目标选择功能，不代表规避或降低反作弊风险。

## 测试

```powershell
python -m unittest discover -s experiments\capture_backends\tests -t . -p "test_*.py" -v
```

单元测试不启动真实游戏。实机截图属于单独的冒烟和能力矩阵测试。
