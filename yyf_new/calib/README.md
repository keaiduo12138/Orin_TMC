# 天眸 ↔ RealSense 双目标定系统

## 目录结构

```
yyf/calib/
├── README.md                          # 本文档
├── utils.py                          # 共享工具（角点检测、棋盘格参数、I/O）
├── run_calibration.py                 # 一键运行完整流程（推荐入口）
│
├── crop_color_frames.py              # Step 0: RS Color 640×480 → 640×320 裁剪
├── calibrate_tianmou_intrinsics.py    # Step 1: 天眸内参标定（支持多数据集联合）
├── calibrate_realsense_cropped.py     # Step 2: RS 内参标定（裁剪后，支持多数据集联合）
├── calibrate_stereo_extrinsics.py     # Step 3: 双目外参标定（支持多数据集联合）
├── validate.py                        # Step 4: 验证 + 综合评分报告
│
├── export_visible_frames.py           # 预处理：从 bag/tianmou 目录导出帧
├── compare_realsense_intrinsics.py   # 工具：RS 出厂 vs 实测内参对比
├── depth_reprojector.py              # 工具：用外参将 RS 深度投影到天眸视角
│
└── calibration_output/                # 输出目录（运行后自动生成）
    ├── intrinsic_tianmou.json
    ├── intrinsic_realsense_cropped.json
    ├── extrinsic_tianmou_realsense.json
    ├── reproj_error_plot.png
    └── validation/
        ├── validation_result.json
        └── validation_report.png
```

---

## 1. 数据文件特点

### 1.1 图像规格

| 项目 | 天眸 | RealSense Color |
|---|---|---|
| 原始分辨率 | 640 × 320 | 640 × 480 |
| 预处理 | 水平镜像（`flipCode=1`） | 裁剪：顶部 80px + 底部 80px → **640 × 320** |
| 用途 | 内参标定、外参标定 | 内参标定、外参标定 |

### 1.2 天眸镜像

天眸镜头硬件安装方向导致图像左右镜像（`TM_FLIP_HORIZONTAL = True`），**所有涉及天眸图像的处理（内参标定、外参标定）都会自动先做水平镜像**，无需手动预处理。

### 1.3 RealSense 裁剪说明

出厂内参是针对 640×480 的，裁剪后（顶部减 80px）主点 `cy` 会发生变化，因此：
- **出厂设置只能是参考**，不能直接用于裁剪后图像的外参标定
- 必须使用 `calibrate_realsense_cropped.py` 实测裁剪后图像的内参
- `--rs-source calibrated`（默认）：外参标定使用裁剪后实测内参

### 1.4 棋盘格参数

| 参数 | 值 |
|---|---|
| 内角点列数 `BOARD_COLS` | 11 |
| 内角点行数 `BOARD_ROWS` | 8 |
| 棋盘格方格尺寸 `SQUARE_SIZE` | **29.66 mm** |
| 角点检测放大倍率 | 2.0× |
| 亚像素精细化 | `cv2.cornerSubPix`，criteria: EPS + MAX_ITER, 100, 1e-6 |

> **注意**：`SQUARE_SIZE` 必须与采集时实际棋盘格方格尺寸一致，否则内参和基线距离会有系统性偏差。

### 1.5 天眸参考内参（640×320, v2 镜头）

```
fx = 713.01,  fy = 712.73,  cx = 224.18,  cy = 269.00
```
标定结果会与该参考值做 Frobenius 偏差对比，>30% 时会有警告。

---

## 2. 标定流程概述

### 完整流水线（4步）

```
数据准备
  │
  ├── 天眸预处理图像（640×320，已镜像）──────────┐
  └── RS Color 原始（640×480） ──→ [Step 0 裁剪] ─→ 640×320
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Step 1: 天眸内参标定          │
                              │  cv2.calibrateCamera          │
                              │  → intrinsic_tianmou.json      │
                              └──────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Step 2: RS 内参标定（裁剪后） │
                              │  cv2.calibrateCamera          │
                              │  → intrinsic_realsense_cropped.json │
                              └──────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Step 3: 双目外参标定           │
                              │  cv2.stereoCalibrate           │
                              │  → extrinsic_tianmou_realsense.json │
                              └──────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Step 4: 验证 + 评分报告       │
                              │  极线几何 / E矩阵秩 / 物理布局 │
                              │  → validation/validation_result.json │
                              └──────────────────────────────┘
```

### 外参含义

```
P_tianmou = R @ P_realsense + T
```
即：`T` 是 RS 原点在 TM 坐标系中的位置（mm）。

物理布局（参考）：
- `T_y ≈ +91 mm`：RS 在 TM 上方约 9.1 cm
- `|T| ≈ 120 mm`：双目基线约 12 cm

---

## 3. 模式一：单数据集标定（单次采集）

适合只有一组数据的场景。

### 3.1 推荐方式：`run_calibration.py`（一键）

> 若 RS 图像已是 640×320（已裁剪），跳过 `--rs-raw-dirs`：

```bash
cd yyf/calib/

# RS 图像为 640×480 原始 → 自动裁剪后标定
python3 run_calibration.py \
    --tm-data-dirs /path/to/dataset_A/tianmou/ \
    --rs-data-dirs /path/to/dataset_A/color/ \
    --rs-raw-dirs  /path/to/dataset_A/color/ \
    --sample-steps 5 \
    -o ./calibration_output

# RS 图像已为 640×320 → 直接标定（跳过裁剪）
python3 run_calibration.py \
    --tm-data-dirs /path/to/dataset_A/tianmou/ \
    --rs-data-dirs /path/to/dataset_A/color_cropped/ \
    --sample-steps 5 \
    -o ./calibration_output
```

### 3.2 分步执行（独立运行各脚本）

```bash
# Step 0: 裁剪（若需要）
python3 crop_color_frames.py \
    --input /projects/calib_data/output_0327_1624/color \
    --output /projects/calib_data/output_0327_1624/color_cropped/ \
    --crop-top 80 --crop-bottom 80

# Step 1: 天眸内参
python3 calibrate_tianmou_intrinsics.py \
    --data-dirs /projects/calib_data/output_0327_1624/tianmou \
    --output ./calibration_output_1624

# Step 2: RS 内参（裁剪后）
python3 calibrate_realsense_cropped.py \
    --data-dirs /projects/calib_data/output_0327_1624/color_cropped/  \
    --output ./calibration_output

# Step 3: 双目外参
python3 calibrate_stereo_extrinsics.py \
    --tm-data-dirs /path/to/tianmou/ \
    --rs-data-dirs /path/to/color_cropped/ \
    --tm-intrinsic ./calibration_output/intrinsic_tianmou.json \
    --rs-source calibrated \
    --sample-steps 5 \
    -o ./calibration_output

# Step 4: 验证
python3 validate.py \
    --tm-data-dirs /path/to/tianmou/ \
    --rs-data-dirs /path/to/color_cropped/ \
    --extrinsic ./calibration_output/extrinsic_tianmou_realsense.json \
    --output ./calibration_output/validation
```

---

## 4. 模式二：多数据集联合标定（同一安装、两段采集）

适合同一相机安装位置、分两次采集的场景。两个数据集共用**同一套内参**和**同一套外参**，数据越多精度越高。

### 4.1 推荐方式：`run_calibration.py`（一键）

```bash
# 两组数据，每组每5帧取1
python3 run_calibration.py \
    --tm-data-dirs /path/to/ds_A/tianmou/ /path/to/ds_B/tianmou/ \
    --rs-data-dirs /path/to/ds_A/color/   /path/to/ds_B/color/ \
    --rs-raw-dirs  /path/to/ds_A/color/   /path/to/ds_B/color/ \
    --sample-steps 5 5 \
    -o ./calibration_output_joint
```

### 4.2 各数据集独立控制抽样步长

```bash
# 数据集A每5帧取1，数据集B每8帧取1
python3 run_calibration.py \
    --tm-data-dirs /path/to/ds_A/tianmou/ /path/to/ds_B/tianmou/ \
    --rs-data-dirs /path/to/ds_A/color/   /path/to/ds_B/color/ \
    --rs-raw-dirs  /path/to/ds_A/color/   /path/to/ds_B/color/ \
    --sample-steps 5 8 \
    -o ./calibration_output_joint
```

### 4.3 分步执行

```bash
# Step 1: 天眸内参（多数据集联合）
python3 calibrate_tianmou_intrinsics.py \
    --data-dirs /path/to/ds_A/tianmou/ /path/to/ds_B/tianmou/ \
    --sample-steps 5 8 \
    --output ./calibration_output_joint

# Step 2: RS 内参（多数据集联合，裁剪后）
python3 calibrate_realsense_cropped.py \
    --data-dirs /path/to/ds_A/color/ /path/to/ds_B/color/ \
    --sample-steps 5 8 \
    --output ./calibration_output_joint

# Step 3: 双目外参（多数据集联合）
python3 calibrate_stereo_extrinsics.py \
    --tm-data-dirs /path/to/ds_A/tianmou/ /path/to/ds_B/tianmou/ \
    --rs-data-dirs /path/to/ds_A/color/   /path/to/ds_B/color/ \
    --tm-intrinsic ./calibration_output_joint/intrinsic_tianmou.json \
    --rs-source calibrated \
    --sample-steps 5 8 \
    -o ./calibration_output_joint

# Step 4: 验证（自动从 extrinsic JSON 读取目录）
python3 validate.py \
    --extrinsic ./calibration_output_joint/extrinsic_tianmou_realsense.json \
    --output ./calibration_output_joint/validation
```

---

## 5. 常用参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--sample-steps` | 每个数据集的抽样步长（n帧取1）。可各不同，也可共用一个值 | `5` |
| `--max-frames-per-dataset` | 每个数据集最多使用帧数（防内参标定过慢） | `300` |
| `--output`, `-o` | 输出目录 | `./calibration_output` |
| `--skip-tm-intrinsic` | 跳过天眸内参（复用已有 JSON） | — |
| `--skip-rs-intrinsic` | 跳过 RS 内参（复用已有 JSON） | — |
| `--skip-validation` | 跳过验证步骤 | — |
| `--rs-source` | RS 内参来源：`factory`/`calibrated`/`default` | `calibrated` |
| `--tm-intrinsic` | 指定天眸内参 JSON，或用 `ref` 使用 TM_K_REF | — |

---

## 6. 输出文件说明

| 文件 | 说明 |
|---|---|
| `intrinsic_tianmou.json` | 天眸内参：K (3×3), dist, 重投影误差, 帧数, 数据集数 |
| `intrinsic_realsense_cropped.json` | RS 内参（裁剪后 640×320）：K, dist, 与出厂值偏差对比 |
| `extrinsic_tianmou_realsense.json` | 双目外参：R, T, E（归一化奇异值=[1,1,0]）, 基线距离, 逐帧误差 |
| `reproj_error_plot.png` | 逐帧重投影误差柱状图 |
| `validation/validation_result.json` | 综合评分：极线精度 / E矩阵秩 / 物理布局 / 总分 |
| `validation/validation_report.png` | 验证可视化报告图 |
| `visualization/ds*_corners_*.png` | 角点可视化图像（带数据集前缀） |

### extrinsic JSON 关键字段

```json
{
  "R": [[...], [...], [...]],       // P_tm = R @ P_rs + T
  "T": [tx, ty, tz],               // mm，RS原点在TM坐标系中的位置
  "E": [[...], [...], [...]],      // 归一化Essential矩阵
  "baseline_mm": 120.2,            // |T| 基线距离 mm
  "mean_reproj_error": 0.42,       // 平均重投影误差 px
  "mean_epipolar_error_px": 0.38,   // 平均对极距离 px（<1px=优秀）
  "n_valid_frames": 48,
  "n_datasets": 2,
  "calibration_method": "cv2.stereoCalibrate (multi-dataset joint)"
}
```

---

## 7. 验收标准

| 指标 | 优秀 | 良好 | 不合格 |
|---|---|---|---|
| 平均对极距离 | < 1.0 px | 1.0 ~ 2.0 px | > 2.0 px |
| E 矩阵秩约束 (σ₃/σ₁) | < 1% | 1% ~ 5% | > 5% |
| T_y 物理布局 | 50 ~ 200 mm（RS在TM上方） | — | < 0（反向）或 > 200 mm |
| 综合评分 | ≥ 80/100 | 60 ~ 80/100 | < 60/100 |

---

## 8. 常见问题

**Q: 报错 "未找到天眸图像"**
→ 检查文件命名格式：应为 `tianmou_NNNN.png` 或 `NNNN_tianmou.png`，放在 `--tm-data-dirs` 目录或其 `tianmou/` 子目录下。

**Q: RS 内参和出厂值偏差很大（>10%）**
→ 可能是棋盘格尺寸设置错误（`SQUARE_SIZE`），或图像未正确裁剪到 640×320。

**Q: 外参标定 T_y 为负值**
→ 检查坐标系定义：当前定义为 `P_tianmou = R @ P_realsense + T`，T_y 负值表示 RS 在 TM "下方"，需检查相机安装方向。

**Q: 多数据集联合标定精度反而下降**
→ 检查两个数据集是否确实是同一安装（相机相对位置未变化）。若安装有调整，应分别标定各外参后再融合。
