
  python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1723 \
  -b /projects/cxr_data/2026-3-24/17-23-30.bag \
  -o /projects/calib_data/0325_11 \
  --tianmou-range 90:110 \
  --color-range 20:40

  python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1723 \
  -b /projects/cxr_data/2026-3-24/17-23-30.bag \
  -o /projects/calib_data/0325_11 \
  --tianmou-range 90:110 \
  --color-range 20:40

python3 yyf/calib/calibrate_stereo_extrinsics.py \
  --data-dir /projects/calib_data/0325_calibration \
  --tm-intrinsic ./calibration_neican/intrinsic_tianmou.json \
  --rs-source factory \
  --output ./calibration_neican


  /projects/calib_data/0325_calibration/
├── 0325_1_color.png   + 0325_1_tianmou.png  (同一位置的标定对)
├── 0325_2_color.png   + 0325_2_tianmou.png
├── ...
└── 0325_16_color.png + 0325_16_tianmou.png


# 单区间（原有写法）
--tianmou-range 200:400 --color-range 100:150

# 多区间（新增）
--tianmou-range "20:24;30:35;58:65" --color-range "20:24;30:35;58:65"

# Step 1: 天眸内参
python3 yyf/calib/calibrate_tianmou_intrinsics.py \
  --data-dir /projects/calib_data/0325_neican \
  --output ./calibration_neican

# Step 2: RS 内参（可选 --bag）
python3 yyf/calib/compare_realsense_intrinsics.py \
  --data-dir /projects/calib_data/0325_neican \
  --bag /projects/cxr_data/2026-3-24/17-26-26.bag \
  --output ./calibration_neican


  天眸内参标定
============================================================
  图像尺寸: 640×320
  使用帧数: 300

  标定结果 (分辨率 640×320):
  K:
  [ 713.0134  0.0000  224.1754 ]
  [ 0.0000  712.7342  269.0033 ]
  [ 0.0000  0.0000  1.0000 ]
  dist: [-0.332616, 0.277191, -0.000732, -0.000533, -0.249764]

  平均重投影误差: 1.1888 pix

============================================================
  与参考内参 TM_K_REF 对比
============================================================
  TM_K_REF:
  [ 462.1590  0.0000  307.0336 ]
  [ 0.0000  462.4046  149.1036 ]
  [ 0.0000  0.0000  1.0000 ]

  Frobenius 偏差: 52.0%
  偏差 > 30%%，建议检查标定质量或棋盘格尺寸设置

============================================================
  各帧棋盘格距离（辅助验证）
============================================================
     帧 |       tx       ty       tz |     |t| mm
  --------------------------------------------------
     0 |   -187.3   -287.9   1004.6 |    1062
     5 |   -185.2   -284.8   1005.5 |    1061
    10 |   -181.1   -286.8   1015.9 |    1071




    RealSense 出厂内参
============================================================
  从 bag 读取: /projects/cxr_data/2026-3-24/17-26-26.bag
  图像尺寸: 640×480
  K_factory:
  [ 385.4384  0.0000  329.5430 ]
  [ 0.0000  384.5369  241.9413 ]
  [ 0.0000  0.0000  1.0000 ]
  dist_factory: [-0.054241, 0.063625, -0.000543, -0.000278, -0.020249]

============================================================
  检测 RealSense 棋盘格角点（全量）
============================================================
[1/3] 全量检测: 100%|█| 1578/1578 [03:44<00:00,  7.04帧/s]
  全量帧: 1578, 检测成功: 1371
  均匀采样: 300 帧（覆盖 1774348397716~1774348447711）
[2/3] 采样帧检测: 100%|█| 300/300 [00:42<00:00,  7.03帧/s]
  最终有效帧: 300/300

============================================================
  RealSense 棋盘格实测内参标定
============================================================
  使用帧数: 300
  平均重投影误差: 0.6152 pix
  K_calib:
  K_calib:
  [ 386.9580  0.0000  329.2932 ]
  [ 0.0000  386.8793  242.3404 ]
  [ 0.0000  0.0000  1.0000 ]
  dist_calib: [-0.057433, 0.097076, -0.000818, -0.000879, -0.095312]

============================================================
  出厂 vs 实测内参对比
============================================================
  K Frobenius 偏差:    0.42%
  fx 偏差:             0.39%
  fy 偏差:             0.61%
  cx 偏差:             0.25 px
  cy 偏差:             0.40 px
  k1 偏差:             0.0032
  k2 偏差:             0.0335
  k3 偏差:             0.0751

============================================================
  推荐方案
============================================================
  ★ 推荐: 使用出厂内参（bag 读取或 D455 默认值）
    原因: 即使实测结果良好，也推荐使用出厂内参（Intel 工厂校准更权威）

============================================================
  畸变系数健康检查
============================================================
  出厂: k1=-0.0542, k2=+0.0636, p1=-0.0005, p2=-0.0003, k3=-0.0202
  实测: k1=-0.0574, k2=+0.0971, p1=-0.0008, p2=-0.0009, k3=-0.0953
  ✓ 畸变系数在正常范围内

============================================================
  各帧棋盘格距离（辅助验证）
============================================================
  帧1774348397716: |t|=   1155mm  (tx=  116.5 ty=  -17.8 tz= 1149.0)
  帧1774348397848: |t|=   1142mm  (tx=  116.2 ty=  -17.7 tz= 1136.4)
  帧1774348398013: |t|=   1118mm  (tx=  112.4 ty=  -14.7 tz= 1112.4)
  帧1774348398145: |t|=   1092mm  (tx=  114.9 ty=    1.4 tz= 1085.6)
  帧1774348398475: |t|=   1053mm  (tx=  131.8 ty=   23.0 tz= 1044.2)
  帧1774348398607: |t|=   1046mm  (tx=  13


阶段一：内参标定（每台相机独立标定）
Step 1 — 天眸内参标定
标定天眸相机的内参矩阵 K 和畸变系数 dist。
cd /home/nvidia/Desktop/cxr_multi_sensor

python3 yyf/calib/calibrate_tianmou_intrinsics.py \
  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \
  --output ./calibration_output
cd /home/nvidia/Desktop/cxr_multi_sensorpython3 yyf/calib/calibrate_tianmou_intrinsics.py \  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \  --output ./calibration_output
输出文件： ./calibration_output/intrinsic_tianmou.json
关键参数说明：
棋盘格规格：11×8（BOARD_COLS=11, BOARD_ROWS=8），在 utils.py 中定义
棋盘格方格尺寸：29.66 mm（SQUARE_SIZE），必须与采集标定板时的实际尺寸一致
天眸分辨率：640×320
最多使用 300 帧（MAX_INTRINSIC_FRAMES=300）
Step 2 — RealSense 内参对比（可选，推荐执行）
对比 Intel 出厂校准内参与棋盘格实测内参，推荐使用更可靠的那套。
python3 yyf/calib/compare_realsense_intrinsics.py \
  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \
  --bag /projects/cxr_data/2026-3-24/17-26-26.bag \
  --output ./calibration_output
python3 yyf/calib/compare_realsense_intrinsics.py \  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \  --bag /projects/cxr_data/2026-3-24/17-26-26.bag \  --output ./calibration_output
> 如果没有对应的 bag 文件，可以省略 --bag 参数，脚本会使用 D455 默认出厂值。
输出文件： ./calibration_output/compare_realsense_intrinsics.json
推荐逻辑： 代码默认推荐使用出厂内参（Intel 工厂校准精度 ~1%，通常优于棋盘格实测），除非偏差超过阈值。
阶段二：外参标定（双目之间的相对位姿）
Step 3 — 双目外参标定（天眸 ↔ RealSense）
这是核心步骤，通过同时检测两台相机视野中的棋盘格，建立两个相机坐标系之间的旋转矩阵 R 和平移向量 T。
python3 yyf/calib/calibrate_stereo_extrinsics.py \
  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \
  --tm-intrinsic ./calibration_output/intrinsic_tianmou.json \
  --rs-source factory \
  --output ./calibration_output
python3 yyf/calib/calibrate_stereo_extrinsics.py \  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \  --tm-intrinsic ./calibration_output/intrinsic_tianmou.json \  --rs-source factory \  --output ./calibration_output
关键参数：
--tm-intrinsic：天眸内参 JSON 路径（Step 1 的输出）
--rs-source：RealSense 内参来源，factory（推荐）/calibrated/default
--select-frames：交互式选择帧（按空格跳过，q 退出）
--bad-threshold：误差阈值，默认 1.0px，超出标记为坏帧
输出文件： ./calibration_output/extrinsic_tianmou_realsense.json
外参含义：
P_tianmou = R @ P_realsense + T
P_tianmou = R @ P_realsense + T
即：T 是 RS 相机原点在 TM 坐标系中的位置（单位 mm）。
Step 4 — 验证（可选但强烈推荐）
验证外参的精度和物理合理性。
python3 yyf/calib/validate.py \
  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \
  --extrinsic ./calibration_output/extrinsic_tianmou_realsense.json \
  --output ./calibration_output/validation
python3 yyf/calib/validate.py \  --data-dir /projects/calib_data/0324_calibration_depth_16_debug \  --extrinsic ./calibration_output/extrinsic_tianmou_realsense.json \  --output ./calibration_output/validation
验证内容：
对极距离误差（< 1px = 优秀，< 2px = 良好）
本征矩阵 E 的秩约束（σ₃/σ₁ < 1%）
物理布局检查（RS 应在 TM 上方约 100mm）
各帧棋盘格距离（30cm ~ 2m 为合理范围）

