# Whimbox 地图基准、格式与定位机制

## 1. 文档范围

本文分析 Whimbox 2.5.4 的地图基础机制，重点回答以下问题：

1. Whimbox 的地图以什么坐标为基准；
2. 地图资产是什么文件格式；
3. 游戏截图如何与离线地图对齐；
4. 地图如何参与定位、传送、路线录制和自动跑图；
5. 当前地图系统包含什么、不包含什么。

鼠标拖动大地图和旋转镜头的输入实现另见：

- [Whimbox鼠标拖拽与镜头旋转机制](Whimbox鼠标拖拽与镜头旋转机制.md)

完整自动跑图状态机另见：

- [地图定位与自动跑图](../analysis/09-地图定位与自动跑图.md)
- [Whimbox自动跑图视觉闭环](../../../concepts/game_ai_planning/12-Whimbox自动跑图视觉闭环.md)
- [游戏内地图自动采集、拼接与关系网方案](../../../concepts/game_ai_planning/13-游戏内地图自动采集拼接与关系网方案.md)
- [分层关系网与复合能力子图构想](../../../concepts/interface_discovery/分层关系网与复合能力子图构想.md)
- [游戏瞬态故障判读与置信度隔离构想](../../../concepts/interface_discovery/游戏瞬态故障判读与置信度隔离构想.md)
- [AzurLaneAutoScript可借鉴机制与设计思想](../../../references/alas/AzurLaneAutoScript可借鉴机制与设计思想.md)

## 2. 核心结论

Whimbox 运行时真正使用的地图基准是：

> 离线全尺寸地图原图的像素坐标 `PngMapPx`。

它不是以 1920×1080 游戏截图坐标为运行时地图基准，也不是始终直接使用游戏世界坐标。

地图系统本质上是：

```text
静态二维地图PNG
+
实时截图模板匹配
+
人工录制路线点
```

它不是：

- 游戏内部地图 API；
- GIS 或瓦片地图；
- 三维地形地图；
- 碰撞地图；
- 导航网格；
- 自动生成道路的关系图。

地图 PNG 的主要作用是回答“角色当前位于哪里”。至于“应该怎么走”，主要由人工录制的路线 JSON 提供。

## 3. 地图基准：PngMapPx

### 3.1 坐标原点和方向

`PngMapPx` 可以理解为原始全尺寸地图 PNG 上的像素坐标：

```text
原点：(0, 0)，位于图片左上角
X轴：向右增加
Y轴：向下增加
```

运行时以下数据通常都使用 `PngMapPx`：

- 角色当前位置 `position`；
- 大地图当前中心 `bigmap_position`；
- 路线目标点；
- 传送点位置；
- 两点距离；
- 目标方向角；
- 卡住检测位置。

`MiniMap.position` 和 `BigMap.bigmap_position` 的定义分别见：

- [`MiniMap`](../../../../../reference_repos/whimbox/whimbox/map/detection/minimap.py#L11)
- [`BigMap`](../../../../../reference_repos/whimbox/whimbox/map/detection/bigmap.py#L11)

### 3.2 全尺寸地图不一定直接保存在仓库中

仓库当前保存的是全尺寸原图生成的缩小灰度资产，而不是完整彩色大图。但代码中的 `PngMapPx` 仍按照概念上的全尺寸原图计算。

例如奇迹大陆的运行时资产为：

```text
0.5倍灰度图：   11776 × 11264
0.125倍灰度图：  2944 × 2816
概念全尺寸原图：23552 × 22528
```

在 0.5 倍资产上匹配到坐标后，代码除以 0.5，恢复成全尺寸 `PngMapPx`；在 0.125 倍资产上匹配后，则除以 0.125。

## 4. Whimbox 中的坐标体系

Whimbox 地图相关代码同时出现四类坐标。理解地图机制时必须区分它们。

| 坐标名称 | 基准 | 主要用途 |
| --- | --- | --- |
| `GameLoc` | 游戏原生世界二维坐标 | 路线 JSON、传送点 JSON 持久化 |
| `PngMapPx` | 全尺寸地图 PNG 像素 | 运行时定位、目标、距离、方向 |
| `InGameMapPx` | 游戏大地图最大缩放时的渲染尺度 | 把地图差值换成拖动或点击偏移 |
| 1920 逻辑屏幕坐标 | 归一化游戏客户区截图 | UI 裁剪、鼠标位置与最终点击 |

### 4.1 GameLoc

`GameLoc` 是路线文件和传送点文件中使用的游戏原生二维坐标。

例如 [`checkpoints.json`](../../../../../reference_repos/whimbox/whimbox/assets/checkpoints.json) 中保存的传送点位置：

```json
{
  "map": "miraland",
  "type": "teleporter",
  "name": "搭配师协会前",
  "position": [-13172.34765625, -54273.6171875]
}
```

这些数值不能直接与地图图片像素或屏幕像素计算距离。

### 4.2 GameLoc 与 PngMapPx 的转换

转换函数位于 [`map/convert.py`](../../../../../reference_repos/whimbox/whimbox/map/convert.py#L19)：

```text
PngMapPx = GameLoc × (2 / 90) + 地图专属偏移

GameLoc = (PngMapPx - 地图专属偏移) ÷ (2 / 90)
```

`2 / 90` 等于 `1 / 45`，也就是大约每 45 个游戏世界坐标单位对应一个全尺寸地图像素。

各地图偏移定义在 [`detection/cvars.py`](../../../../../reference_repos/whimbox/whimbox/map/detection/cvars.py#L24)：

| 地图 | GameLoc 原点对应的 PngMapPx |
| --- | --- |
| 奇迹大陆 `miraland` | `(16799, 15723)` |
| 星海 `starsea` | `(2448, 1051)` |
| 家园 `home` | `(1989, 1243)` |
| 万相境 `wanxiang` | `(1950, 987)` |

因此，同一个 `GameLoc` 数值放在不同 `map_name` 下，会转换为不同的地图图片位置。路线和位置数据必须同时携带地图名称。

### 4.3 InGameMapPx

`InGameMapPx` 这个名称容易被误解。它不是游戏世界坐标，而是大地图界面处于最大缩放时的地图渲染尺度。

转换关系见 [`convert_InGameMapPx_to_PngMapPx()`](../../../../../reference_repos/whimbox/whimbox/map/convert.py#L7)：

```text
PngMapPx = InGameMapPx × BIGMAP_POSITION_SCALE
InGameMapPx = PngMapPx ÷ BIGMAP_POSITION_SCALE
```

不同地图使用不同经验比例：

| 地图 | `BIGMAP_POSITION_SCALE` |
| --- | ---: |
| `miraland` | 0.637 |
| `starsea` | 0.62 |
| `home` | 0.61 |
| `wanxiang` | 0.62 |

这些参数见 [`detection/cvars.py`](../../../../../reference_repos/whimbox/whimbox/map/detection/cvars.py#L61)。实际代码主要使用两个位置的差值，而不是把绝对 `InGameMapPx` 直接当成屏幕点击坐标。

### 4.4 1920 逻辑屏幕坐标

地图定位结果最终需要驱动鼠标时，才转换到 1920×1080 逻辑截图坐标。例如拖动大地图、点击传送点时，会将目标和当前中心的 `PngMapPx` 差值换算成大地图界面偏移，再叠加逻辑屏幕中心 `(960, 540)`。

因此不能直接进行以下计算：

```text
PngMapPx + 屏幕坐标
```

必须先经过地图显示比例换算。

## 5. 地图资产格式

### 5.1 地图文件位置

地图图片位于：

```text
whimbox/assets/imgs/Maps/
```

资源名称通过 [`imgs_index.json`](../../../../../reference_repos/whimbox/whimbox/assets/imgs/imgs_index.json) 映射到实际文件路径。

地图资产在 [`map_assets.py`](../../../../../reference_repos/whimbox/whimbox/map/detection/map_assets.py#L41) 中按地图名称分组。

### 5.2 三类主要地图文件

每张受支持地图通常包含三类派生 PNG：

| 文件后缀 | 内容 | 用途 |
| --- | --- | --- |
| `*_luma_05x.png` | 原图 0.5 倍亮度/灰度图 | 小地图局部定位 |
| `*_luma_0125x.png` | 原图 0.125 倍亮度/灰度图 | 大地图全局定位 |
| `*_mask_0125x.png` | 黑白或索引色有效区域掩膜 | 排除不可匹配区域 |

亮度图加载后是二维 `uint8` 数组，每个像素表示 0～255 的亮度值。掩膜加载后同样作为二维 `uint8` 数组使用。

它们不是携带道路、节点、地形高度等属性的数据文件，只是供 OpenCV 进行模板匹配的栅格图像。

### 5.3 当前地图资产尺寸

| 地图 | 0.5 倍图 | 0.125 倍图 | 概念全尺寸 |
| --- | ---: | ---: | ---: |
| 奇迹大陆 | 11776×11264 | 2944×2816 | 23552×22528 |
| 星海 | 2560×2048 | 640×512 | 5120×4096 |
| 家园 | 2560×1536 | 640×384 | 5120×3072 |
| 万相境 | 4096×3072 | 1024×768 | 8192×6144 |

文件名中的 `v11`、`v2`、`v1` 等字段体现了地图资源版本，但当前代码是静态指定文件名，没有运行时自动协商地图版本的机制。

### 5.4 资产生成方式

开发工具 [`map_assets_gen.py`](../../../../../reference_repos/whimbox/whimbox/dev_tool/map_assets_gen.py#L8) 从完整地图原图生成运行时资产：

```text
完整彩色地图原图
→ 应用有效区域mask
→ 转换为亮度图
→ 缩放到0.5倍
→ 保存为*_luma_05x.png

完整彩色地图原图
→ 转换为亮度图
→ 缩放到0.125倍
→ 保存为*_luma_0125x.png
```

缩放采用最近邻插值。完整彩色原图主要用于开发阶段生成派生资产，运行时不需要保留。

### 5.5 方向箭头资产

地图目录还包含：

```text
ArrowRotateMap.png
ArrowRotateMapAll.png
```

它们不是地理地图，而是预先生成的人物方向箭头旋转图集，用于把小地图中的人物箭头匹配成角度。生成逻辑见 [`gen_ArrowRotateMap()`](../../../../../reference_repos/whimbox/whimbox/dev_tool/map_assets_gen.py#L49)。

## 6. 地图资产如何加载

[`MapAsset`](../../../../../reference_repos/whimbox/whimbox/map/detection/utils.py#L16) 的加载过程是：

```text
资产名称
→ imgs_index.json查找相对路径
→ Pillow读取PNG
→ 转换为NumPy数组
→ 保存在MapAsset.img
```

[`map_assets.py`](../../../../../reference_repos/whimbox/whimbox/map/detection/map_assets.py#L41) 在模块导入时构造各地图的资产对象，因此地图数组会提前进入内存，而不是每次定位时重新读取磁盘。

`MAP_ASSETS_DICT` 按运行时地图名称组织资产：

```python
MAP_ASSETS_DICT[map_name] = {
    "luma_05x": ...,
    "luma_0125x": ...,
    "mask_0125x": ...,
}
```

## 7. 如何确定当前使用哪张地图

首次初始化时，[`update_region_and_map_name()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L107) 会：

```text
打开大地图
→ OCR识别当前区域名称
→ 区域名称映射为map_name
→ 从MAP_ASSETS_DICT选择对应地图资产
```

区域到地图资产的映射位于 [`detection/cvars.py`](../../../../../reference_repos/whimbox/whimbox/map/detection/cvars.py#L13)。例如多个奇迹大陆区域会共同映射到 `miraland`。

当前需要注意：[`trans_region_name_to_map_name()`](../../../../../reference_repos/whimbox/whimbox/map/detection/utils.py#L6) 对明确不支持区域返回 `unsupported`，但对无法识别的未知文字默认返回 `home`。这可能把 OCR 失败误判为家园地图。

## 8. 首次全局定位：大地图

### 8.1 初始化流程

[`reinit_smallmap()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L120) 执行：

```text
进入游戏大地图
→ OCR识别区域
→ 选择离线地图资产
→ 将大地图放大到最大
→ 截取完整游戏大地图画面
→ 在0.125倍全局地图上匹配
→ 得到大地图视野中心PngMapPx
→ 用该位置初始化小地图跟踪器
```

大地图负责提供一个能够覆盖整个地图的全局初始位置。

### 8.2 大地图模板匹配

[`BigMap._predict_bigmap()`](../../../../../reference_repos/whimbox/whimbox/map/detection/bigmap.py#L22) 的主要过程为：

1. 将当前大地图截图转换为亮度图；
2. 根据地图的显示比例和 `0.125` 搜索比例缩小截图；
3. 在整张 `luma_0125x` 地图中执行 `cv2.matchTemplate()`；
4. 对相关矩阵进行高斯差分，突出局部峰；
5. 使用 `mask_0125x` 排除不可匹配区域；
6. 对峰值附近进行三次插值细化；
7. 加上截图中心偏移；
8. 除以 `0.125`，恢复到全尺寸 `PngMapPx`。

结果保存为：

- `bigmap_position`；
- `bigmap_similarity`；
- `bigmap_similarity_local`。

这一步回答的是：

> 当前游戏大地图画面的中心，对应离线全尺寸地图的哪个像素位置？

## 9. 连续局部定位：左上角小地图

### 9.1 为什么不持续搜索全图

每帧在完整地图中搜索小地图截图会消耗大量计算，并容易在相似地形中出现多个候选峰。因此 Whimbox 在完成全局初始化后，只搜索上次位置附近。

### 9.2 小地图截取区域

1920×1080 逻辑截图中的小地图参数定义在 [`detection/cvars.py`](../../../../../reference_repos/whimbox/whimbox/map/detection/cvars.py#L35)：

```text
小地图中心：(181, 122)
小地图半径：102
位置匹配半径：100
```

[`MiniMap._get_minimap()`](../../../../../reference_repos/whimbox/whimbox/map/detection/minimap.py#L36) 从完整游戏截图裁取这一区域。

### 9.3 环形遮罩

人物方向箭头位于小地图中心，会干扰地图纹理匹配。Whimbox 使用一个环形掩膜：

```text
保留小地图外围地形
排除圆外区域
排除中心人物箭头区域
```

生成逻辑见 [`create_minimap_mask()`](../../../../../reference_repos/whimbox/whimbox/map/detection/map_assets.py#L8)。

### 9.4 局部匹配流程

[`MiniMap._predict_position()`](../../../../../reference_repos/whimbox/whimbox/map/detection/minimap.py#L43) 执行：

```text
当前小地图截图
→ 转亮度图
→ 按地图比例和0.5搜索比例缩放
→ 以上一次PngMapPx为中心
→ 从luma_05x裁取约1.3倍小地图大小的邻域
→ 使用环形mask进行模板匹配
→ 查找相关峰并插值细化
→ 除以0.5恢复PngMapPx
```

最终更新：

- `position`；
- `position_similarity`；
- `position_similarity_local`。

### 9.5 位置跳变过滤

[`verify_position()`](../../../../../reference_repos/whimbox/whimbox/map/detection/minimap.py#L131) 根据两次定位间隔和预计最大移动速度判断候选是否合理：

```text
允许距离约为：MOVE_SPEED × 时间间隔 + 1
```

超出范围的突然跳变通常会被拒绝。若超过 20 秒没有接受新位置，则下一次候选会直接通过，以允许长暂停后的恢复。

## 10. 地图如何参与路线执行

### 10.1 路线文件保存GameLoc

路线 JSON 的点位保存为游戏原生 `GameLoc`。保存路线时，[`RecordPathTask.save_path()`](../../../../../reference_repos/whimbox/whimbox/task/navigation_task/record_path_task.py#L102) 会：

```text
录制时的PngMapPx
→ 转换为GameLoc
→ 写入路线JSON
```

这样路线文件不直接依赖某张缩小地图资产的像素尺寸。

### 10.2 执行时统一转为PngMapPx

[`AutoPathTask.__init__()`](../../../../../reference_repos/whimbox/whimbox/task/navigation_task/auto_path_task.py#L42) 加载路线后，会深拷贝路线点并执行：

```text
路线点GameLoc
→ 根据路线info.map转换
→ PngMapPx
```

之后当前角色位置和所有路线目标都在同一坐标系中，可以直接计算：

- 当前位置到目标的距离；
- 下一个目标方向角；
- 是否到达目标；
- 是否长时间没有移动；
- 是否需要传送到附近传送点。

### 10.3 地图只负责定位，不负责生成路线

自动跑图并不会分析地图图片中的道路并自动生成路径。路线依然是一组人工录制的顺序点：

```text
点0 → 点1 → 点2 → 点3 → ……
```

地图图片使系统能够判断“当前到达哪个位置”，路线点则告诉系统“接下来应该向哪里移动”。

## 11. 地图如何参与传送

### 11.1 传送点数据

[`checkpoints.json`](../../../../../reference_repos/whimbox/whimbox/assets/checkpoints.json) 保存传送点名称、区域、地图名称和 `GameLoc`。

模块加载时，[`nikki_teleporter.py`](../../../../../reference_repos/whimbox/whimbox/map/data/nikki_teleporter.py#L18) 将所有传送点转换为 `PngMapPx`。因此角色位置、目标点和传送点可以直接在同一坐标系中比较距离。

### 11.2 拖动并点击传送点

传送时的主要过程：

```text
目标PngMapPx
→ 找到附近最近传送点PngMapPx
→ 打开并最大化大地图
→ 全局识别当前大地图中心PngMapPx
→ 计算当前中心与传送点的地图差值
→ 换算成InGameMapPx拖动量
→ 拖动地图
→ 再次识别并修正
→ 换算为1920屏幕点击位置
→ 点击传送图标
```

主要实现见 [`Map._move_bigmap()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L201) 和 [`bigmap_tp()`](../../../../../reference_repos/whimbox/whimbox/map/map.py#L333)。

## 12. 地图包含什么、不包含什么

### 12.1 地图当前包含的信息

地图资产能够提供：

- 二维视觉纹理；
- 地图有效匹配区域；
- 角色二维位置；
- 人物朝向；
- 镜头朝向；
- 传送点二维坐标；
- 路线点二维坐标。

### 12.2 地图当前不包含的信息

地图资产本身不包含：

- 道路拓扑；
- 可行走区域；
- 碰撞边界；
- 斜坡、悬崖和高度；
- 楼层关系；
- 动态障碍；
- 地图对象语义；
- 自动寻路代价；
- 节点与边的关系网。

所以 Whimbox 无法仅凭这张地图图片执行通用 A* 或导航网格寻路。可走路线来自人工录制，而不是从图片自动推导。

## 13. 当前实现的明显限制

### 13.1 地图是静态视觉资产

游戏更新导致地图纹理、比例、区域范围或 UI 显示方式变化时，需要重新生成或校准地图资产。文件名虽然带有版本字段，但没有自动检测游戏地图版本的机制。

### 13.2 小地图定位依赖上一次位置

小地图只在上次位置附近搜索。发生远距离传送、地图识别错误或位置状态丢失时，需要重新打开大地图进行全局初始化。

### 13.3 未知区域可能误判为家园

区域 OCR 无法映射时，当前代码默认返回 `home`，而不是显式的 `unknown`。这可能使后续视觉匹配加载错误的地图资产。

### 13.4 大地图匹配分数没有硬性拒绝阈值

大地图定位记录了整体和局部相似度，但 [`update_bigmap()`](../../../../../reference_repos/whimbox/whimbox/map/detection/bigmap.py#L69) 没有根据最低置信阈值拒绝候选。错误峰值可能直接成为新的初始位置。

### 13.5 二维位置不能表达立体世界

相同平面位置上的不同高度、楼层或地下区域可能无法仅依靠二维 `PngMapPx` 区分。当前路线和业务状态必须通过额外动作或页面判断弥补。

## 14. 对后续关系网系统的参考

Whimbox 地图适合被理解为“空间定位证据层”，而不是完整规划层。新系统可以继承它的静态地图先验、全局初始化、局部跟踪和执行后复核思想，但不能把 Whimbox 专用的 `PngMapPx`直接提升为所有游戏的统一坐标，也不能把地图节点和可复用操作模块塞进同一棵层级树。

### 14.1 `PngMapPx`是 Whimbox 适配器坐标

本文前面所有 `PngMapPx`结论仍然针对 Whimbox 2.5.4：它以概念全尺寸地图 PNG 左上角为原点，并通过固定资产比例与 `GameLoc`转换。

在多游戏系统中，应明确区分：

| 概念 | 适用范围 | 含义 |
| --- | --- | --- |
| `GameLoc` | 特定游戏适配器 | 游戏自身或外部工具提供的世界坐标 |
| `PngMapPx` | Whimbox地图资产 | 某套完整地图PNG上的像素坐标 |
| `LocalMapPx` | 新系统自建地图 | 某个 `MapSpace`内部的局部连续坐标 |
| 屏幕逻辑坐标 | 某次窗口与截图环境 | UI裁剪和最终输入定位 |

如果导入 Whimbox 地图数据，应原样标记：

```text
coordinate_type: PngMapPx
map_asset_set_version: whimbox_miraland_v11
adapter_id: whimbox_infinity_nikki
```

如果系统通过自动拼接地图建立自己的坐标，则使用：

```text
coordinate_type: LocalMapPx
map_space_id: miraland_main
```

二者之间只有在已经拟合并验证 `CoordinateTransform`时才能转换：

```text
CoordinateTransform
├── transform_id / transform_revision
├── source_space / source_coordinate_type
├── target_space / target_coordinate_type
├── transform_model / parameters
├── control_points[]
├── residual_statistics
├── valid_region
├── map_asset_versions[]
└── verified_at
```

不能仅因为两套坐标都以像素表示，就直接相加、比较距离或复用经验比例。

### 14.2 使用 `MapSpace + LocalMapPx`表达分层世界

`MapSpace`表示一套内部坐标连续、能够使用同一定位方法解释的位置域：

```text
MapSpace: miraland_main
MapSpace: miraland_underground_01
MapSpace: dungeon_012_floor_1
MapSpace: dungeon_012_floor_2
MapSpace: home
```

同一个平面位置上的不同楼层不应强行叠进一张二维图。它们可以各自拥有 `LocalMapPx`，再通过离散空间边连接：

```text
地面入口
-> 进入能力
-> 地下MapSpace入口节点

一楼电梯
-> 乘坐电梯能力
-> 二楼电梯出口节点

传送锚点A
-> 传送能力
-> 另一个MapSpace的锚点B
```

`MapSpace`不是必须固定成一棵行政区域树。一个副本入口可能被多个任务、区域或通用系统引用；空间组织可以是带跨层连接的图，运行时根据当前位置和目标选择相关子图。

### 14.3 空间图与能力模块DAG分开

新系统至少需要两个相互引用、但职责不同的结构：

```text
SpatialGraph
├── MapSpace
├── PlaceNode
├── PortalNode
├── RouteEdge
└── SpatialConstraint

CapabilityModuleDAG
├── 打开地图
├── 拖动地图ContinuousController
├── 选择传送锚点
├── 执行传送CompositeCapability
├── 录制路线执行器
├── 楼层切换能力
└── 定位恢复能力
```

二者分别回答：

| 结构 | 回答的问题 |
| --- | --- |
| 空间图 | 地点在哪里、哪些地点相连、连接条件是什么 |
| 能力模块DAG | 这条边实际由哪些可复用观察、输入、验证和恢复模块完成 |

空间边只保存 `executor_ref`或能力契约，不复制完整操作流程：

```text
RouteEdge
├── from_node / to_node
├── context_requirements
├── executor_ref
├── expected_outcome_ref
├── route_evidence_refs[]
└── edge_revision
```

例如“打开地图并传送”的能力可以被多个锚点和区域引用；地图拖动控制器也可以被不同 `MapSpace`通过各自 `CalibrationProfile`复用。它们是多父引用的模块DAG，不应因为出现在多个区域就复制多份实现和统计。

计划执行时，由主能力图选择空间目标和关系边，再延迟展开 `executor_ref`对应的模块版本。运行时调用栈负责子能力返回原调用者，空间图本身不承担线程调度或输入清理。

### 14.4 静态先验与动态观察融合

可以借鉴 Whimbox 和 ALAS 的共同思想：已有地图资产、地点和路线是静态先验，当前截图只是一批带来源和置信度的动态观察。

```text
MapAssetSet / MapTile / Landmark / RouteEdge
-> 提供先验候选

新鲜FrameRecord
-> 定位器产生候选位置和原始分数

上下文、运动约束和历史位置
-> 筛选或保留多个候选

LocalizationDecision
-> 保存候选集合、accepted标记和专业reason_code
```

一次截图错配、游戏Bug、地图图标缺失或捕获旧帧不能直接覆盖已有地图知识。遇到冲突时应先判断：

1. 当前帧是否新鲜且来源正确；
2. 地图、区域和楼层上下文是否一致；
3. 是否只是角色、时间、活动或游戏版本造成的视觉变体；
4. 候选移动是否符合时间和速度约束；
5. 是否需要保留多个位置假设；
6. 是否应触发大地图全局重定位或请求用户确认。

小地图局部跟踪失败时，旧位置可以作为先验保留，但不能伪装成当前已验证位置。全局重定位成功后产生新的 `LocalizationDecision`，而不是回写修改旧证据。

### 14.5 地图和变换必须版本化

后续地图数据至少需要记录：

```text
game_version
map_space_revision
map_asset_set_version
tile_revision
landmark_detector_version
coordinate_transform_revision
localization_policy_version
route_edge_revision
controller_revision
calibration_profile_id
graph_snapshot_id
```

这些版本解决的问题不同：

- 游戏版本决定地图和UI是否可能发生变化；
- 地图资产版本决定像素坐标基于哪套图片；
- `MapSpace`修订描述空间边界和内部坐标定义；
- 坐标变换版本描述两套坐标如何对齐；
- 定位器版本决定候选和分数如何生成；
- 路线边版本描述前置条件、执行器和后置验证；
- `CalibrationProfile`只描述当前环境下连续输入的响应参数。

计划开始时应固定不可变的地图、关系图、检测器和执行器版本。运行过程中发现新地图块或新路线时，可以生成候选修订，但不能静默改变已经批准计划所使用的空间定义。

自动采集地图的具体分块、拼接、锚点识别和分阶段实现见[游戏内地图自动采集、拼接与关系网方案](../../../concepts/game_ai_planning/13-游戏内地图自动采集拼接与关系网方案.md)。本文只负责说明它与 Whimbox 坐标及定位机制之间的边界。

### 14.6 定位、关系和环境指标隔离

一次地图定位不应只输出一个无法解释的总置信度。建议分别维护：

| 指标 | 含义 |
| --- | --- |
| `registration_quality` | 地图块或截图之间的几何配准质量 |
| `localization_confidence` | 当前帧位于某个坐标候选的可信程度 |
| `landmark_identity_confidence` | 锚点、入口或地点身份是否明确 |
| `spatial_relation_correctness` | 两个空间节点之间的关系是否真实存在 |
| `conditional_execution_reliability` | 在指定上下文中执行路线边的成功率 |
| `current_availability` | 路线、入口或传送当前是否可用 |
| `perception_health` | 截图、地图资产和定位器是否有效 |
| `recovery_success_rate` | 全局重定位、返回锚点等恢复方法是否有效 |

这些指标不能直接相加。特别是：

```text
position_similarity低
!= 路线关系错误

输入送错窗口
!= 地图坐标错误

按钮临时变灰或游戏Bug
!= 传送边不存在

游戏版本不兼容
!= 历史路线样本应被删除
```

只有当前置空间和游戏上下文成立、定位有效、环境健康、输入正确送达且新鲜后置证据仍明确未到达目标时，才适合降低路线边的条件执行可靠性。地图资产错误、定位器失败和环境故障应进入各自指标或隔离样本。

### 14.7 `LocalizationDecision`不是公共 `ExecutionOutcome`

定位器只输出位置候选、几何分数、约束证据和 `LOCALIZATION_MATCHED`、`NO_LOCALIZATION_MATCH`、`AMBIGUOUS_CANDIDATES`等专业 `reason_code`，不能自行定义另一套业务结果。公共终态统一使用：

```text
ExecutionOutcome = SUCCEEDED / FAILED / UNKNOWN / CANCELLED / BLOCKED
```

`BLOCKED`表示输入前已经明确知道空间能力当前不可执行，并且没有发送业务输入。地图版本静态不兼容、路线或入口当前不可用、必要定位门禁不满足时，应优先返回 `BLOCKED`；如果路线输入已经发出但后置位置无法确认，结果仍为 `UNKNOWN`。

一次统一结果还要携带 `reason_code + FailureAttribution + SampleDisposition`：

| 场景 | `ExecutionOutcome` | `reason_code` | `FailureAttribution` | `SampleDisposition` |
| --- | --- | --- | --- | --- |
| 新鲜定位证据确认到达目标空间节点 | `SUCCEEDED` | `TARGET_LOCATION_CONFIRMED` | - | `VALID_SUCCESS` |
| 输入已发出，但后置定位只有冲突候选 | `UNKNOWN` | `AMBIGUOUS_CANDIDATES` | `PERCEPTION_FAILURE` | `PERCEPTION_INVALID` |
| 输入已发出，后置帧是缓存旧帧 | `UNKNOWN` | `STALE_FRAME` | `AUTOMATION_ENVIRONMENT` | `QUARANTINED` |
| 环境健康且定位可靠，有限执行后明确未到达 | `FAILED` | `TARGET_NOT_REACHED` | `EXECUTOR_FAILURE` | `VALID_CAPABILITY_FAILURE` |
| 地图资产与当前游戏版本静态不兼容，未发送输入 | `BLOCKED` | `MAP_VERSION_INCOMPATIBLE` | `GAME_VERSION_INCOMPATIBLE` | `QUARANTINED` |
| 路线、入口或传送当前明确不可用，未发送输入 | `BLOCKED` | `SPATIAL_CAPABILITY_UNAVAILABLE` | `EXPECTED_UNAVAILABLE` | `EXPECTED_UNAVAILABLE` |
| 定位前置门无法满足且未发送业务输入 | `BLOCKED` | `LOCALIZATION_PRECONDITION_UNMET` | `PERCEPTION_FAILURE` | `PERCEPTION_INVALID` |
| 游戏出现已知瞬态地图Bug，导致已执行输入的结果无法判断 | `UNKNOWN` | `KNOWN_BUG_SIGNATURE` | `GAME_CLIENT_TRANSIENT` | `TRANSIENT_ENVIRONMENT_FAULT` |
| 游戏出现已知瞬态地图Bug，且新鲜证据明确命中失败结果 | `FAILED` | `KNOWN_BUG_SIGNATURE` | `GAME_CLIENT_TRANSIENT` | `TRANSIENT_ENVIRONMENT_FAULT` |
| 用户取消且路线worker、按键和资源完成清理 | `CANCELLED` | `USER_CANCELLED` | - | `USER_CANCELLED` |

定位器可以立即提供专业 `reason_code`，但 `FailureAttribution`由统一证据归因流程生成，`SampleDisposition`再决定样本进入定位器、地图版本、路线能力、环境故障还是隔离统计。恢复成功只表示重新获得可定位、可继续规划的状态，不能覆盖原 StepAttempt 的 `BLOCKED`、`FAILED`或 `UNKNOWN`。

`RecoveryRun`始终独立保存。全局重定位、返回锚点或重新加载地图成功后，必须新建 StepAttempt；不能把恢复结果与原路线 Attempt 压缩成一次成功。

### 14.8 推荐的定位证据对象

一次定位输出可以表示为：

```json
{
  "decision_id": "localization_0001842",
  "frame_id": 1842,
  "application_id": "infinity_nikki",
  "window_id": "infinity_nikki.main",
  "window_generation": 12,
  "map_space_id": "miraland_main",
  "coordinate_type": "PngMapPx",
  "position": [16506.3, 14516.9],
  "source": "minimap_template_match",
  "detector_version": 3,
  "map_asset_set_version": "whimbox_miraland_v11",
  "similarity": 0.91,
  "local_similarity": 0.08,
  "candidate_count": 2,
  "motion_constraint_passed": true,
  "accepted": true,
  "reason_code": "LOCALIZATION_MATCHED"
}
```

这里仍保留 Whimbox 的原始 `PngMapPx`含义。如果经过坐标适配器转换到自建地图，应生成带 `transform_revision`的新派生证据，不覆盖原始定位记录。

### 14.9 最终数据流

综合后的关系网定位流程为：

```text
WindowRegistry和CaptureRouter提供新鲜FrameRecord
-> 定位器在固定MapAssetSet版本上生成位置候选
-> 上下文与运动约束形成LocalizationDecision
-> CoordinateTransform按需转换到目标MapSpace坐标
-> SpatialGraph选择地点节点和空间边
-> CapabilityModuleDAG展开executor_ref
-> 原子能力或ContinuousController执行输入闭环
-> 新鲜后置定位验证目标节点
-> Step输出公共ExecutionOutcome
-> 定位器记录专业reason_code
-> 统一生成FailureAttribution与SampleDisposition
-> 地图、关系、环境和恢复指标按SampleDisposition分别更新
```

最终应把 Whimbox 的 `PngMapPx + 静态地图匹配`保留为一个优秀的游戏专用定位适配器；在其上增加 `MapSpace + LocalMapPx`、空间图、能力模块DAG、版本快照、公共 `ExecutionOutcome`和指标隔离，才能扩展为多区域、多楼层、多游戏的关系网系统。
