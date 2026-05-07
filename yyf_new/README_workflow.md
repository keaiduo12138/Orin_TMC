# 三路传感器数据对齐与导出流程

> 天眸事件相机 (TianMou) + RealSense D455 Color + RealSense D455 Depth 三路同步导出

---

## 流程总览

```
Step 1: post_process.py       →  生成索引 JSON（三路时间对齐）
Step 2: export_aligned_frames.py →  导出图像文件（原始 PNG）
Step 3: depth_rec.py           →  深度图重投影 + 生成 frames.json（时空对齐）
```

---

## Step 1 — 生成三路对齐索引

```bash
conda activate tianmoucv

python3 yyf/post_process.py \
    --tianmou-dir  /projects/cxr_data/19700101_0806 \
    --bag          /projects/cxr_data/1970-1-1/8-6-50.bag \
    --temporal     /projects/cxr_data/temporal_files/tianmou_timestamp_19700101_080719.csv \
    -o output_1970.json
```

**输出**: `output_1970.json` —— 包含每帧的 `tianmou_idx`、`depth_counter`、`color_counter` 等

| 参数 | 说明 |
|------|------|
| `--tianmou-dir` | 天眸数据目录（从 `start_auto_gui.sh` 录制得到） |
| `--bag` | RealSense bag 文件路径 |
| `--temporal` | 天眸时间戳 CSV 文件 |
| `-o` | 输出的索引 JSON 文件名 |

---

## Step 2 — 导出原始图像

```bash
# 1. 创建输出目录（需要 sudo 权限）
sudo mkdir -p /projects/calib_data/output_1970
sudo chmod 777 /projects/calib_data/output_1970

# 2. 导出图像（必须加 --use-raw-depth 使用原始 Z16 深度）
python3 yyf/export_aligned_frames.py \
    -i  output_1970.json \
    -t  /projects/cxr_data/19700101_0806 \
    -b  /projects/cxr_data/1970-1-1/8-6-50.bag \
    -o  /projects/calib_data/output_1970 \
    --use-raw-depth
```

**输出目录结构**:
```
output_1970/
├── tianmou/         天眸 RGB 图像   (320×640, uint8)
├── color/           RealSense Color (640×480, uint8)
├── depth/           RealSense Depth (640×480, uint16 Z16)
├── comparison/      三路并排对比图
└── realsense_raw/   中间临时目录（可忽略）
```

---

## Step 3 — 深度图重投影 + 生成索引

```bash
python3 yyf/calib/depth_rec.py \
    --base   /projects/calib_data/output_1970 \
    --index  output_1970.json
```

**说明**:
- 读取 `depth/` 目录下的原始深度图
- 使用标定好的外参（RS→TM）将深度反投影到天眸坐标系
- 输出镜像后的 Z16 深度图和并排对比图
- **自动生成** `depth_tm/frames.json` 侧文件索引

**输出目录结构**:
```
output_1970/
├── depth/           ← Step 2 导出（原始 RS 深度）
├── tianmou/         ← Step 2 导出（天眸 RGB）
├── depth_tm/        ← Step 3 输出（重投影后的对齐深度）
│   ├── depth_0000.png   对齐后的 Z16 深度图 (320×640)
│   ├── depth_0001.png
│   ├── ...
│   └── frames.json     侧文件索引（关键！）
└── compare_0327/    并排对比图（仅可视化用途）
```

---

## depth_tm/frames.json 详解

这是下游使用最重要的索引文件，每条记录的字段：

```json
{
  "seq_idx":      "0000",       # PNG 文件序号（对应 depth_0000.png）
  "tianmou_idx":  123,          # 天眸原始帧号
  "depth_counter": 1001,         # RealSense Depth Frame Counter
  "color_counter": 2001,        # RealSense Color Frame Counter
  "depth_idx":    0,            # Depth 帧序号
  "dt_ms":        33.3          # 与锚定点的时间差（ms）
}
```

**使用场景**: 下游模型读取 `depth_tm/depth_0000.png` 后，查 `frames.json` 得到对应的 `tianmou_idx`，再去找天眸复杂数据文件（时间差分、空间差分、体素网格等）。

---

## 完整一次性运行（参考命令）

```bash
conda activate tianmoucv
cd /home/nvidia/Desktop/cxr_multi_sensor

# ========== 替换以下 4 个变量 ==========
TIANMOU_DIR="/projects/cxr_data/20260327_1624"
BAG_FILE="/projects/cxr_data/2026-3-27/16-24-48.bag"
TEMPORAL_FILE="/projects/cxr_data/temporal_files/tianmou_timestamp_20260327_162556.csv"
OUTPUT_NAME="output_0327_1624"
OUTPUT_DIR="/projects/calib_data/output_0327_1624"
# ========================================

# Step 1: 生成索引
python3 yyf/post_process.py \
    --tianmou-dir "$TIANMOU_DIR" \
    --bag          "$BAG_FILE" \
    --temporal     "$TEMPORAL_FILE" \
    -o "${OUTPUT_NAME}.json"

# Step 2: 导出图像
sudo mkdir -p "$OUTPUT_DIR" && sudo chmod 777 "$OUTPUT_DIR"
python3 yyf/export_aligned_frames.py \
    -i  "${OUTPUT_NAME}.json" \
    -t  "$TIANMOU_DIR" \
    -b  "$BAG_FILE" \
    -o  "$OUTPUT_DIR" \
    --use-raw-depth

# Step 3: 重投影 + 生成 frames.json
python3 yyf/calib/depth_rec.py \
    --base  "$OUTPUT_DIR" \
    --index "${OUTPUT_NAME}.json"
```

---

## 目录与文件说明

| 路径 / 文件 | 用途 |
|------------|------|
| `depth/` | RealSense 原始深度，640×480，Z16 mm |
| `tianmou/` | 天眸 RGB，320×640，与天眸坐标系一致 |
| `depth_tm/` | **重投影后的对齐深度**，320×640，Z16 mm，与天眸空间对齐 |
| `depth_tm/frames.json` | **核心索引**，把 PNG 文件对应回原始帧信息 |
| `compare_0327/` | 深度 vs 天眸可视化对比图，仅人工检查用 |
| `comparison/` | Step 2 的三路对比图，天眸 + Color + Depth 并排 |

---

## 常见问题

**Q: `--use-raw-depth` 必须加吗？**
> 必须。`depth_rec.py` 里的标定参数是基于 Z16 原始深度校准的，处理后深度会导致重投影结果不准确。

**Q: `depth_rec.py` 不加 `--index` 可以吗？**
> 可以，但不会生成 `frames.json`，只有 `depth_tm/` 下的 PNG 图像。

**Q: 两个外参版本（`calib/depth_rec.py` vs `align_depth_0326.py`）选哪个？**
> 用 `calib/depth_rec.py`，它使用了最新的正确标定参数（`R_inv`、`T_inv`）。

**Q: 如何检查重投影结果？**
> 查看 `compare_0327/` 下的对比图，左侧是深度 jet，右侧是天眸原图，深度轮廓应与天眸图像边缘对齐。
