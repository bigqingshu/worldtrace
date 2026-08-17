# Whimbox 鼠标拖拽与镜头旋转机制

## 1. 文档范围

本文只分析 Whimbox 2.5.4 中与鼠标滑动直接相关的两条控制链：

1. 在大地图界面按住左键拖动地图；
2. 在游戏主界面水平移动鼠标旋转镜头。

键盘移动、页面跳转、传送业务流程和完整自动寻路不在本文展开。相关总体背景可参阅：

- [平台抽象与交互核心](../analysis/07-平台抽象与交互核心.md)
- [地图定位与自动跑图](../analysis/09-地图定位与自动跑图.md)
- [Whimbox键鼠执行](../../../concepts/game_ai_planning/05-Whimbox键鼠执行.md)
- [Whimbox任务调度停止与资源互斥](../../../concepts/game_ai_planning/06-Whimbox任务调度停止与资源互斥.md)
- [Whimbox错误恢复与执行证据](../../../concepts/game_ai_planning/10-Whimbox错误恢复与执行证据.md)
- [基于用户游玩监控的界面与交互要素发现构想](../../../concepts/interface_discovery/基于用户游玩监控的界面与交互要素发现构想.md)
- [鼠标滚轮缩放与视口控制构想](../../../concepts/interface_discovery/鼠标滚轮缩放与视口控制构想.md)
- [多窗口输入归属与游戏视角异常恢复构想](../../../concepts/interface_discovery/多窗口输入归属与游戏视角异常恢复构想.md)
- [游戏瞬态故障判读与置信度隔离构想](../../../concepts/interface_discovery/游戏瞬态故障判读与置信度隔离构想.md)
- [AzurLaneAutoScript可借鉴机制与设计思想](../../../references/alas/AzurLaneAutoScript可借鉴机制与设计思想.md)

## 2. 核心结论

Whimbox 没有把“大地图拖动”和“镜头旋转”实现成一次性、绝对精确的鼠标动作，而是分别构造了两个视觉反馈闭环：

```text
大地图：地图坐标误差 → 鼠标拖动距离 → 重新截图定位 → 继续修正
镜头：目标角度误差 → 鼠标水平位移 → 重新识别角度 → 继续修正
```

两条链最终都使用平台输入接口注入真实鼠标事件，但控制含义不同：

| 场景 | 鼠标动作 | 视觉反馈 | 主要状态输出 |
| --- | --- | --- | --- |
| 拖动大地图 | 绝对移动到中心、左键按下、平滑相对移动、左键松开 | 大地图与全局地图资产的模板匹配 | `bigmap_position`、相似度、最终点击坐标 |
| 旋转镜头 | 不按鼠标键，只发送水平相对位移 | 小地图中镜头视野扇形的角度识别 | `rotation`、`rotation_confidence` |

## 3. 鼠标输入基础层

### 3.1 调用分层

上层统一通过 [`InteractionBGD`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_core.py#L28) 调用鼠标操作：

```text
地图/寻路代码
→ itt.move_to()、itt.left_down()、itt.left_up()
→ InteractionNormal
→ 平台 InputManager
→ Windows win32api.mouse_event / SetCursorPos
```

[`InteractionBGD.move_to()`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_core.py#L422) 会把当前截图对象记录的原始游戏分辨率传给交互实现。操作装饰器还会检查游戏窗口是否在前台，失去焦点时尝试恢复前台窗口，见 [`before_operation()`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_core.py#L294)。

### 3.2 Windows 底层事件

Windows 实现位于 [`platform/windows/input.py`](../../../../../reference_repos/whimbox/whimbox/platform/windows/input.py#L18)：

```python
# 左键按下和松开
win32api.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
win32api.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

# 相对移动
win32api.mouse_event(MOUSEEVENTF_MOVE, dx, dy)

# 绝对设置系统光标位置
win32api.SetCursorPos((x, y))
```

因此它不是向游戏内部调用“拖动地图”或“转动镜头”API，而是模拟系统鼠标输入。游戏必须处于能够接收这些事件的状态。

macOS 走同一个上层接口，但底层改用 Quartz `CGEvent`，见 [`platform/macos/input.py`](../../../../../reference_repos/whimbox/whimbox/platform/macos/input.py#L47)。

### 3.3 1920 逻辑坐标到物理坐标

Whimbox 的截图和视觉资产以宽度 1920 为统一逻辑坐标。发送鼠标输入时，再根据游戏客户区原始宽度换算：

```text
缩放比例 = 原始客户区宽度 / 1920
物理移动量 = 1920逻辑移动量 × 缩放比例
```

例如游戏客户区为 1280×720：

```text
逻辑移动 300 像素
→ 物理移动 300 × 1280 / 1920
→ 实际发送 200 像素
```

对应代码见 [`InteractionNormal.move_to()`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_normal.py#L103)。绝对移动还会把游戏客户区坐标转换成屏幕坐标；16:10 等较高画面会根据顶部、底部或中心锚点修正纵向位置。

### 3.4 平滑相对移动

平滑移动由 [`smooth_move_relative()`](../../../../../reference_repos/whimbox/whimbox/interaction/interaction_normal.py#L75) 实现：

```text
距离 = √(dx² + dy²)
步数 = max(2, int(距离 / 5))
单步位移 = 总位移 / 步数
单步延时 = 总持续时间 / 步数
```

默认总持续时间为 0.2 秒。它本质上是均匀分段移动，不是贝塞尔曲线或加减速曲线。

## 4. 大地图拖动机制

### 4.1 为什么拖动前固定地图缩放

[`maximize_bigmap_scale()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L184) 会先确认当前处于大地图页面，再重复点击地图缩放按钮，直到识别到最大缩放特征，最多尝试三次。

这样做是为了让“地图资产坐标差”和“鼠标拖动像素”保持相对稳定的换算关系。如果地图缩放层级变化，同样的鼠标位移会对应完全不同的地图距离。

### 4.2 获取当前地图中心

[`get_bigmap_posi()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L193) 会获取完整游戏截图，然后调用大地图识别：

```text
大地图截图
→ 转灰度亮度图
→ 按地图配置缩放
→ 与完整地图灰度资产做模板匹配
→ 查找最大相关峰
→ 局部插值细化
→ 得到当前大地图中心在地图PNG中的坐标
```

识别实现位于 [`BigMap._predict_bigmap()`](../../../../../reference_repos/whimbox/whimbox/map/detection/bigmap.py#L22)，更新以下状态：

- `bigmap_position`：当前视野中心在全局地图 PNG 中的坐标；
- `bigmap_similarity`：整体模板相关峰；
- `bigmap_similarity_local`：高斯差分后的局部峰值。

### 4.3 计算拖动距离

主要参数在 [`Map.__init__()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L25)：

```python
MAP_POSI2MOVE_POSI_RATE = 0.6
BIGMAP_TP_OFFSET = 20
BIGMAP_MOVE_MAX = 300
TP_RANGE = 200
```

拖动量计算公式为：

```python
dx = (当前地图X - 目标地图X) * 0.6
dy = (当前地图Y - 目标地图Y) * 0.6
```

随后分别将 `dx` 和 `dy` 限制在 `[-300, 300]`。完整实现见 [`_move_bigmap()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L201)。

这里有意使用“当前坐标减目标坐标”。例如目标位于当前中心右侧 500 个地图像素：

```text
dx = (当前 - 目标) × 0.6
   = -500 × 0.6
   = -300
```

鼠标向左拖动 300 个逻辑像素，地图画面向左移动，视野中心相应向目标的右侧方向推进。

### 4.4 实际拖动动作

Whimbox 没有使用单独的高级 `drag()` 方法，而是手动组合基础事件：

```python
itt.move_to([960, 540], anchor=ANCHOR_CENTER)
itt.left_down()
itt.move_to([dx, dy], relative=True, smooth=True)
itt.delay(0.2)
itt.left_up()
```

含义依次是：

1. 将鼠标移动到游戏逻辑画面的中心；
2. 按住鼠标左键；
3. 在保持按下状态时平滑发送相对移动；
4. 等待地图动画或渲染稳定；
5. 松开左键。

### 4.5 拖动后的视觉复核

松开鼠标后，代码再次识别 `bigmap_position`：

- 若目标已经进入可点击的传送范围，返回目标对应的屏幕点击坐标；
- 若地图中心与目标的距离不超过 20，返回屏幕中心；
- 否则递归调用 `_move_bigmap()`，再次截图、计算并拖动。

所以大地图移动是一个分段闭环：

```text
识别中心
→ 最多拖动300逻辑像素
→ 重新识别中心
→ 判断是否到达
→ 未到达则继续
```

调用传送时，[`bigmap_tp()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L333) 使用 `_move_bigmap()` 返回的屏幕点点击传送图标，然后再通过 OCR 判断按钮文字是“传送”还是“追踪”。

## 5. 镜头旋转机制

### 5.1 计算目标角度

自动跑图每轮会根据角色当前位置和下一个目标点计算目标方向角，见 [`inner_step_change_view()`](../../../../../reference_repos/whimbox/whimbox/task/navigation_task/auto_path_task.py#L495)：

```text
当前位置 + 目标位置
→ 计算目标方向角 target_degree
→ 获取当前镜头角度 rotation
→ 计算两者最短角度差
```

若目标距离小于 0.5，代码不旋转镜头；若需要转动至少 45° 且角色正在移动，会先停止前进，避免大角度转向时继续冲错方向。

[`calculate_delta_angle()`](../../../../../reference_repos/whimbox/whimbox/view_and_move/utils.py#L23) 把角度差收束到最短旋转方向。例如当前角度 350°、目标角度 10°时，最终选择约 20°的旋转，而不是反向旋转约 340°。

### 5.2 从小地图识别当前镜头角度

镜头角度不是从游戏内存读取，而是从小地图上的半透明视野区域识别。主要过程位于 [`MiniMap._get_minimap_subtract()`](../../../../../reference_repos/whimbox/whimbox/map/detection/minimap.py#L211) 和 [`MiniMap._predict_rotation()`](../../../../../reference_repos/whimbox/whimbox/map/detection/minimap.py#L265)：

```text
截取圆形小地图
→ 根据当前位置裁取对应地图背景
→ 对齐并扣除背景
→ 保留镜头视野的半透明区域
→ 将圆形图像展开成矩形
→ 用Scharr算子提取水平方向边缘
→ 寻找视野区域左右边缘峰值
→ 换算成镜头角度
```

识别结果写入：

- `rotation`：当前镜头角度；
- `rotation_confidence`：角度峰值的置信程度。

为了减少单帧抖动，[`get_safe_rotation()`](../../../../../reference_repos/whimbox/whimbox/view_and_move/view.py#L14) 会在约 0.4 秒内重复取角度，等待连续结果的差值小于指定阈值。

### 5.3 角度转换为鼠标位移

镜头控制入口 [`direct_cview()`](../../../../../reference_repos/whimbox/whimbox/view_and_move/view.py#L9) 非常直接：

```python
px = angle2movex(angle)
itt.move_to([px, 0], relative=True)
```

它不会按下鼠标左键或右键，而是利用游戏主界面默认的鼠标视角控制模式，直接发送水平相对位移。

角度与鼠标像素的换算保存在内存配置中：

```python
鼠标水平像素 = 旋转角度 × view_rotation_ratio
```

见 [`view_and_move/utils.py`](../../../../../reference_repos/whimbox/whimbox/view_and_move/utils.py#L3)。

### 5.4 自动校准镜头灵敏度

不同游戏灵敏度、显示分辨率和系统鼠标设置会导致“移动多少像素等于旋转多少度”不固定。因此 [`calibrate_view_rotation_ratio()`](../../../../../reference_repos/whimbox/whimbox/view_and_move/view.py#L51) 会执行试转校准：

```text
识别当前镜头角度
→ 按当前比例请求旋转90°
→ 等待0.3秒
→ 再次识别镜头角度
→ 计算实际旋转量
→ 修正 view_rotation_ratio
```

修正公式为：

```python
view_rotation_ratio *= 90 / 实际旋转角度
```

例如请求旋转 90°，实际只旋转 45°：

```text
新比例 = 原比例 × 90 / 45
       = 原比例 × 2
```

下一次请求同样角度时，发送的鼠标水平位移就会扩大一倍。

### 5.5 跑图过程中的闭环修正

[`change_view_to_angle()`](../../../../../reference_repos/whimbox/whimbox/view_and_move/view.py#L69) 每次只执行一次角度读取和一次相对移动；它本身不会在函数内部持续旋转到完全准确。

真正的闭环来自自动跑图循环反复调用视角调整：

```text
获取当前位置
→ 计算目标方向
→ 识别当前镜头方向
→ 转动一次
→ 继续跑图
→ 下一轮再次截图并修正
```

目标角度误差小于默认容差时不再移动鼠标。自动跑图调用处使用的容差为 3°。

## 6. 两种控制方式的差异

| 对比项 | 大地图拖动 | 镜头旋转 |
| --- | --- | --- |
| 是否按住鼠标键 | 按住左键 | 不按鼠标键 |
| 起始位置 | 先绝对移动到画面中心 | 不依赖绝对光标位置 |
| 移动方向 | X、Y二维相对移动 | 只移动X轴 |
| 是否平滑分段 | 是，默认0.2秒 | 否，单次相对事件 |
| 误差来源 | 地图模板定位、缩放比例、拖动响应 | 小地图角度识别、游戏灵敏度 |
| 自适应方式 | 拖后重新定位并递归修正 | 90°试转校准比例并在跑图循环修正 |
| 最终输出 | 可点击的屏幕位置 | 更新后的镜头角度状态 |

## 7. 这样设计的好处

### 7.1 不要求一次鼠标动作绝对准确

地图拖动和镜头旋转都会在后续截图中重新观测真实结果。输入缩放、游戏动画和小幅识别误差可以由下一轮修正吸收。

### 7.2 视觉坐标与物理分辨率解耦

视觉计算统一基于 1920 逻辑宽度，输入层最后才映射到真实客户区像素。多数分辨率适配逻辑不会扩散到地图和寻路算法中。

### 7.3 能适应不同镜头灵敏度

镜头控制通过试转测量真实响应，不把固定的“每度多少像素”写死在配置文件中。

### 7.4 地图移动幅度受限

单次拖动限制在 300 逻辑像素以内，可以减少一次拖动距离过大造成的定位丢失，并给视觉复核留下机会。

## 8. 当前实现的明显风险

### 8.1 左键按下与松开没有 `try/finally`

大地图拖动在 `left_down()` 和 `left_up()` 之间会进行地图定位、计算、鼠标移动和等待。若中间抛出异常，代码不能保证执行 `left_up()`，可能留下系统认为左键仍处于按下状态的问题。

更稳妥的结构应当是：

```python
itt.left_down()
try:
    # 计算并拖动
finally:
    itt.left_up()
```

### 8.2 整次拖动不是原子操作

`operation_lock` 只保护单次 `move_to()` 或滚轮调用，没有覆盖“移动到中心—按下—拖动—松开”的整个序列。若多个任务同时注入输入，其他鼠标或键盘操作可能插入拖动过程。

### 8.3 递归修正没有明确最大次数

`_move_bigmap()` 在未达到目标时递归调用自身。虽然会检查任务停止标志，但没有最大拖动次数。如果地图识别长期错误、窗口没有响应或拖动无效，可能持续递归并最终达到 Python 递归深度限制。

### 8.4 平滑移动存在整数截断

平滑移动把每一步的浮点位移直接转换成整数，但最后没有补偿累计余数。尤其是较短或斜向移动时，实际发送的总位移可能小于请求值。大地图闭环能够继续修正，但会增加拖动次数。

### 8.5 镜头置信度没有参与拒绝判定

小地图识别会生成 `rotation_confidence`，但当前视角调整主要依靠连续角度是否接近来判断稳定，没有依据置信度直接拒绝低质量结果。复杂场景中可能把稳定但错误的角度用于鼠标控制。

### 8.6 校准比例只保存在进程内存

`view_rotation_ratio` 初始值为 1，校准结果不会持久化。程序重启后需要重新校准；游戏灵敏度、分辨率或输入环境在运行中变化时，也没有自动识别配置变化的机制。

## 9. 对后续系统设计的参考

Whimbox 最值得保留的是“请求一个变化量，随后从画面测量实际结果”的控制思想，而不是其中某个固定比例、递归函数或鼠标宏。新系统应把大地图拖动、镜头旋转、滚轮缩放和滑块调节统一收束到 `ContinuousController`，并把一次环境下测得的输入响应保存为独立的 `CalibrationProfile`。

### 9.1 `ContinuousController`表达控制规律

关系网中的 Step 应表达目标，而不是录制好的鼠标轨迹：

```text
错误表达：
移动到(960, 540)
-> 左键按下
-> 向左移动300像素
-> 左键松开

推荐表达：
将大地图中心移动到目标地点附近
将镜头转向目标方向
```

推荐的控制器职责为：

```text
ContinuousController
├── controller_id / controller_revision
├── semantic_target
├── observer_ref
├── actuator_ref
├── active_calibration_profile_ref
├── stable_frame_policy
├── verification_policy
├── iteration_policy
├── safety_policy
└── cleanup_policy
```

两条 Whimbox 控制链可以映射为：

| Whimbox机制 | 新系统中的位置 |
| --- | --- |
| `bigmap_position`模板定位 | 地图拖动控制器的状态观察器 |
| `MAP_POSI2MOVE_POSI_RATE` | 初始响应先验，不是永久真值 |
| 最大300逻辑像素 | 单次动作安全上限 |
| `rotation`与`rotation_confidence` | 镜头控制器的状态值和感知证据 |
| `view_rotation_ratio` | 当前环境下的校准参数 |
| 拖动或旋转后重新截图 | 后置验证器 |

地图位置和镜头角度仍是页面状态携带的连续变量，不应为每个坐标或每一度角建立独立关系节点。关系图只保存带参数的 Step，具体位移由控制器在运行时计算。

### 9.2 `CalibrationProfile`与节点定义分离

`ContinuousController`定义“如何观察、如何控制和如何验证”；`CalibrationProfile`只描述某个具体环境中测得的输入响应：

```text
CalibrationProfile
├── profile_id / profile_revision
├── controller_id / controller_revision
├── application_id / game_version
├── window_size / DPI / ui_scale
├── normalized_resolution
├── capture_backend / input_backend
├── map_zoom_level_or_control_mode
├── formula_type / formula_parameters
├── minimum_effective_delta
├── dead_zone / hysteresis / quantization
├── sample_refs[]
├── residual_statistics
├── verified_at
├── confidence
└── lifecycle_status
```

这三类版本不能混为一谈：

```text
controller_revision
-> 控制算法和验证契约的版本

profile_revision
-> 某种环境指纹下的校准参数版本

StepAttempt
-> 使用某一固定控制器版本和校准版本的一次运行事实
```

以下条件应让活动 `CalibrationProfile`降级或失效，而不是删除整个关系节点：

- 游戏灵敏度、地图缩放等级或控制模式变化；
- 游戏版本、分辨率、窗口尺寸、DPI或UI缩放变化；
- 输入后端或截图后端发生切换；
- 找不到原来的视觉锚点；
- 相同请求量的实际响应持续偏离历史残差范围；
- 响应方向改变、死区扩大或出现新的吸附行为。

已有有效 Profile 时可以跳过完整试转，但不能跳过本次执行后的结果验证。Profile 失效后应保留旧记录供回放和比较，只是不再作为活动公式。

### 9.3 输入事务与桌面级所有权

Whimbox 的 `operation_lock`只保护局部调用。多窗口系统中，物理键鼠必须由全局资源 `desktop_physical_input`串行仲裁，并且锁应覆盖前置确认、完整复合输入和后置验证：

```text
按固定顺序取得game_control:<window_id>
-> 取得desktop_physical_input
-> 建立FocusLease
-> 验证目标窗口代次和focus_epoch
-> 获取新鲜前置帧
-> 执行完整鼠标序列
-> 强制释放按键和鼠标按钮
-> 等待稳定帧
-> 获取后置帧并验证Outcome
-> 保存证据
-> 释放租约
```

`FocusLease`至少需要绑定：

```text
target_application_id
target_window_id
target_window_generation
focus_epoch
acquired_at
verified_at
```

每个复合输入阶段都要确认其 `focus_epoch`仍然有效。若其他窗口抢走焦点，应立即使旧 epoch 失效、停止尚未发送的事件、结束当前 Attempt，并执行统一输入清理。仅调用 `SetForegroundWindow`或 Whimbox 的 `before_operation()`不能证明输入已经由目标游戏消费；仍需用目标窗口和必要的见证窗口响应验证真实输入归属。

左键释放也不能只依赖正常路径。拖动控制器应在 `finally`、取消清理、焦点租约失效和 worker 异常四条路径上调用同一个幂等释放协议。

### 9.4 固定等待升级为稳定帧门

Whimbox 使用固定的 `0.2`秒或 `0.3`秒等待，这在单一已知游戏中简单有效。通用系统应由 `StableFrameGate`判断当前观测是否已经适合验证：

```text
输入结束
-> 等待至少一张来源正确的新帧
-> 检查frame_id和窗口代次
-> 连续观察控制相关区域
-> 位置、角度或配准结果进入稳定范围
-> 输出稳定后置观测
```

稳定性应优先观察与控制目标有关的量，而不是要求整张画面完全不变。例如角色待机动画可以持续变化，但地图中心位置已经稳定；镜头画面仍有粒子效果，但角度估计已经收敛。

`StableFrameGate`必须有最大等待时间。后置帧过期、来源切换或始终无法稳定时，不能用缓存旧帧宣布成功；若输入可能已经生效，公共 `ExecutionOutcome`应为 `UNKNOWN`。

### 9.5 有限迭代和振荡看门狗

后续实现不应继续使用无上限递归，而应采用显式、可取消的有限迭代：

```text
for attempt in 1..max_attempts
    观察当前值
    计算残差
    若达标则结束
    计算受限动作量
    执行动作
    等待稳定并重新观察
    检查进度、振荡和资源状态
```

看门狗至少跟踪：

- 最大尝试次数和总截止时间；
- 单次及累计鼠标输入上限；
- 连续无有效推进次数；
- 残差是否在两个区间之间来回振荡；
- 位置或角度是否突然跳到不可能范围；
- 校准预测误差是否持续扩大；
- `focus_epoch`和截图新鲜度是否发生变化；
- 地图是否到达边界、镜头是否进入特殊控制模式。

地图边界造成的零位移、目标窗口未消费输入和视觉定位失败，表面上都可能是“没有推进”，但故障归因不同，不能统一增加鼠标量继续尝试。

### 9.6 公共 `ExecutionOutcome`与故障归因

连续控制 Step 的公共终态统一使用：

```text
ExecutionOutcome = SUCCEEDED / FAILED / UNKNOWN / CANCELLED / BLOCKED
```

其中 `BLOCKED`表示当前条件已经明确不允许执行，并且尚未发送业务输入。静态版本不兼容、能力当前不可用、无法建立有效 `FocusLease`以及校准前置门不满足，都应优先返回 `BLOCKED`，而不是伪造一次 `FAILED`。如果已经发送业务输入，便不能再用 `BLOCKED`掩盖未知结果。

`CALIBRATION_REQUIRED`、`FOCUS_EPOCH_EXPIRED`、`NO_RESPONSE`等都只是专业 `reason_code`，不是另一套顶层终态。统一结果还要携带经证据验证后生成的 `FailureAttribution`和 `SampleDisposition`：

| 场景 | `ExecutionOutcome` | `reason_code` | `FailureAttribution` | `SampleDisposition` |
| --- | --- | --- | --- | --- |
| 新鲜后置证据明确达到目标 | `SUCCEEDED` | `TARGET_REACHED` | - | `VALID_SUCCESS` |
| 有新鲜证据，有限纠偏后仍未达到目标 | `FAILED` | `TARGET_NOT_REACHED` | `EXECUTOR_FAILURE` | `VALID_CAPABILITY_FAILURE` |
| 输入已经发出，后置截图失败 | `UNKNOWN` | `CAPTURE_FAILED` | `AUTOMATION_ENVIRONMENT` | `TRANSIENT_ENVIRONMENT_FAULT` |
| 输入已经发出，只有缓存旧帧 | `UNKNOWN` | `STALE_FRAME` | `AUTOMATION_ENVIRONMENT` | `QUARANTINED` |
| 目标窗口无预期响应，微量输入明确作用到其他游戏窗口 | `FAILED` | `INPUT_OWNERSHIP_VIOLATION` | `INPUT_ROUTING` | `INPUT_ROUTING_INVALID` |
| 角度或位置识别证据冲突 | `UNKNOWN` | `POSTCONDITION_CONFLICT` | `PERCEPTION_FAILURE` | `PERCEPTION_INVALID` |
| 当前校准失效且未发送业务输入 | `BLOCKED` | `CALIBRATION_INVALID` | `EXPECTED_UNAVAILABLE` | `EXPECTED_UNAVAILABLE` |
| 无法建立有效焦点租约且未发送业务输入 | `BLOCKED` | `FOCUS_EPOCH_EXPIRED` | `INPUT_ROUTING` | `INPUT_ROUTING_INVALID` |
| 游戏版本与控制器静态声明不兼容 | `BLOCKED` | `GAME_VERSION_UNSUPPORTED` | `GAME_VERSION_INCOMPATIBLE` | `QUARANTINED` |
| 用户取消且完成输入释放和worker收尾 | `CANCELLED` | `USER_CANCELLED` | - | `USER_CANCELLED` |

如果输入已经发送而代码随后抛错，不能自动记为 `FAILED`。只要实际游戏结果尚不明确，就应记录 `outcome=UNKNOWN`和专业 `reason_code=WORKER_ERROR`；`FailureAttribution`根据证据选择 `EXECUTOR_FAILURE`或 `UNKNOWN`，`SampleDisposition`使用 `ATTRIBUTION_UNKNOWN`或 `QUARANTINED`，禁止把这次样本直接计入控制关系可靠性。

专业模块可以立即提供 `reason_code`，但 `FailureAttribution`必须由统一归因流程根据窗口、帧、输入和版本证据生成；`SampleDisposition`再决定样本可以更新哪项长期统计。只有在环境指纹有效、校准样本充分时仍反复证明控制公式或前置、动作、Outcome契约本身定义错误，才令 `FailureAttribution=CAPABILITY_DEFINITION_ERROR`、`SampleDisposition=VALID_CAPABILITY_FAILURE`。

`RecoveryRun`始终独立保存。例如校准重建或重新取得焦点成功，只表示恢复动作成功；原来的 `BLOCKED`、`FAILED`或 `UNKNOWN` Attempt 不能被覆盖。恢复后需要创建新的 StepAttempt 再执行控制目标。

### 9.7 指标必须隔离

“鼠标是否真正产生预期效果”应进入执行证据，但不能压缩成一个总置信度。至少分别维护：

| 指标 | 含义 |
| --- | --- |
| `relation_correctness` | 在有效前置条件下，这个控制关系是否成立 |
| `current_availability` | 当前页面、角色状态和游戏规则是否允许控制 |
| `observer_quality` | 地图位置或镜头角度是否被可靠识别 |
| `calibration_fit_quality` | 当前 Profile 对输入响应的预测误差 |
| `input_routing_health` | 输入是否由目标窗口消费 |
| `environment_health` | 窗口、截图、帧新鲜度和游戏进程是否健康 |
| `recovery_success_rate` | 失焦、控制模式异常等恢复能力是否有效 |

只有在前置状态成立、感知有效、环境健康、输入正确送达且新鲜后置证据仍明确失败时，才适合降低关系或控制能力本身的可靠性。焦点错投、黑帧、旧帧、游戏瞬态Bug和校准失效应分别归因。

### 9.8 结构化运行证据

一次控制 Attempt 可以保存：

```json
{
  "attempt_id": "attempt_000218",
  "controller_id": "map.pan",
  "controller_revision": 4,
  "calibration_profile_id": "map.pan.profile.37",
  "target_window_id": "infinity_nikki.main",
  "target_window_generation": 12,
  "focus_epoch": 108,
  "before_frame_id": 18420,
  "requested_target": [1210.8, 731.6],
  "requested_delta": [-300, 80],
  "injected_events": ["move", "left_down", "relative_move", "left_up"],
  "after_frame_id": 18427,
  "observed_before": [1024.5, 780.2],
  "observed_after": [1182.4, 742.1],
  "residual": [28.4, -10.5],
  "attempt_index": 2,
  "outcome": "UNKNOWN",
  "reason_code": "STALE_FRAME",
  "failure_attribution": "AUTOMATION_ENVIRONMENT",
  "sample_disposition": "QUARANTINED",
  "cleanup_complete": true
}
```

这里的 `outcome`是公共 `ExecutionOutcome`中的顶层终态，描述 Step 是否达到目标；`reason_code`保存控制器直接观察到的专业原因；归因、样本处置、位置相似度、输入归属判定、校准残差和清理结果仍保留为独立证据。这样既能回放 Whimbox 式视觉闭环，也不会让某一种局部失败污染整个关系网。

### 9.9 最终控制链

将上述约束合并后，推荐控制链为：

```text
取得控制资源和FocusLease
-> 获取新鲜前置帧
-> ContinuousController读取当前状态
-> 选择或校验CalibrationProfile
-> 计算受限动作量
-> 在有效focus_epoch内执行完整输入事务
-> 幂等释放所有输入
-> StableFrameGate取得新鲜稳定后置帧
-> 验证目标并输出公共ExecutionOutcome
-> 记录专业reason_code
-> 统一生成FailureAttribution与SampleDisposition
-> 按SampleDisposition隔离更新指标
-> 保存Attempt、帧、动作、校准和清理证据
```

Whimbox 的大地图和镜头控制可以作为第一批 `ContinuousController`测试样本：它们已经具备清晰的请求量、视觉状态量和后置反馈，但新系统仍需补上环境指纹、桌面输入所有权、稳定帧门、有限迭代、公共 `ExecutionOutcome`、统一故障归因和完整证据链。
