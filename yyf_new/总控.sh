pkill -f start_auto_gui.sh

python3 yyf/post_process_fast.py \
    --tianmou-dir /projects/cxr_data/20260415_16-13-42 \
    --bag /projects/cxr_data/2026-4-15/16-13-50.bag \
    --temporal /projects/cxr_data/20260415_16-13-42/tianmou_timestamp_20260415_161412.csv \
    -o output_0415_1.json

cd /home/nvidia/Desktop/cxr_multi_sensor/yyf &&
conda activate tianmoucv
python3 yyf/post_process.py \
    --tianmou-dir /projects/cxr_data/20260317_1439 \
    --bag /projects/cxr_data/2026-3-17/14-39-47.bag \
    --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260317_144025.csv \
    -o output_0317_1.json

我想做一个时间精度对齐检测的验证。我有tianmou和realsense两个传感器，这两个传感器都有内部读取时间戳的功能。你可以分别学习
yyf/post_process.py  tianmou数据（需激活tianmoucv这个环境，读取）
yyf/check_bag_metadata.py realsense读取Frame timestamp
根据这两个代码，你尝试写一个在外部时间戳对齐后，读取天眸和realsense各自的内部时间戳的代码。

    cd build
    ls -la ./main
    cmake .. && make

    sudo python3 /home/nvidia/Desktop/cxr_multi_sensor/yyf/power_1v8_ctrl.py on

    python3 yyf/check_bag_metadata.py /projects/cxr_data/2026-3-23/15-48-21.bag



    python3 yyf/export.py \
    --tianmou-dir /projects/cxr_data/20260319_2133 \
    --bag /projects/cxr_data/2026-3-19/21-33-16.bag \
    --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260319_213341.csv \
    -n -8 \
    -e /projects/calib_data/

        python3 yyf/export.py \
    --tianmou-dir /projects/cxr_data/20260319_2133 \
    --bag /projects/cxr_data/2026-3-19/21-33-16.bag \
    --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260319_213341.csv \
    -e /projects/calib_data/20260319



    python3 yyf/post_process.py \
    --tianmou-dir /projects/cxr_data/20260319_2133 \
    --bag /projects/cxr_data/2026-3-19/21-33-16.bag\
    --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260319_213341.csv \
    -o /projects/calib_data/20260319/index.json


python3 yyf/post_process.py     --tianmou-dir /projects/cxr_data/20260320_1
848     --bag /projects/cxr_data/2026-3-20/18-48-10.bag    --tempo
ral /projects/cxr_data/temporal_files/tianmou_timestamp_20260320_185013.csv     -o /projects/calib_data/20260320/index.json

python3 yyf/post_process_color.py \
  --tianmou-dir /projects/cxr_data/20260320_2325 \
  --bag /projects/cxr_data/2026-3-20/23-25-36.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260320_232558.csv \
  -o /projects/calib_data/20260320/index_color.json

  python yyf/record_d455_genlock.py

--------------------------------------------------------------------------------------------

    python3 yyf/check_bag_metadata.py /projects/cxr_data/2026-3-23/18-24-9.bag


  conda activate tianmoucv

python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/20260324_2054 \
  --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260324_205537.csv \
  -o output_calibration_0324depth.json


sudo mkdir -p /projects/calib_data/0324_calibration_depth && sudo chmod 777 /projects/calib_data/0324_calibration_depth
  python3 yyf/export_aligned_frames.py \
  -i output_calibration_0324depth.json \
  -t /projects/cxr_data/20260324_2054 \
  -b /projects/cxr_data/2026-3-24/20-54-59.bag \
  -o /projects/calib_data/0324_calibration_depth

--------------------------------------------------------------------------------------------
python3 yyf/check_bag_metadata.py /projects/cxr_data/2026-3-24/15-23-56.bag

python3 yyf/post_process.py \
  1cxr_data/2026-3-24/15-37-34.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_2026031· 24_153815.csv \
  -o output_calibration_0324_3.json


sudo mkdir -p /projects/calib_data/0324_calibration_v2 && sudo chmod 777 /projects/calib_data/0324_calibration_v2
  python3 yyf/export_aligned_frames.py \
  -i output_calibration_0324_1.json \
  -t /projects/cxr_data/20260324_1530 \
  -b /projects/cxr_data/2026-3-24/15-30-31.bag \
  -o /projects/calib_data/0324_calibration_v2
------------------标定-------------------------------
# 基础用法（使用所有帧，自动采样最多100帧）
python3 stereo_calib.py \
  --data-dir /projects/calib_data/0323_calibration_v1 \
  --bag /projects/cxr_data/2026-3-20/18-48-10.bag \
  --output ./calibration_output

# 指定具体帧（跳过前5后5，均匀采样）
python3 stereo_calib.py \
  --data-dir /projects/calib_data/0323_calibration_v1 \
  --bag /projects/cxr_data/2026-3-20/18-48-10.bag \
  --frames 10 15 20 25 30 35 40 50 60 70 80 \
  --output ./calibration_output

# 跳过天眸内参标定，直接用参考内参（更快）
python3 stereo_calib.py --data-dir /projects/calib_data/0323_calibration_v1 \
  --skip-stage2 --output ./calibration_output

python3 stereo_calib.py \
  --data-dir /projects/calib_data/0323_calibration_v1 \
  --bag /projects/cxr_data/2026-3-20/18-48-10.bag \
  --frames 10 15 20 25 30 35 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500 510 520 530 540 550 560 570 580 590 600 610 620 630 640 650 660 670 680 690 700 710 720 730 740 750 760 770 780 790 800 810 820 830 840 850 860 870 880 890 900 910 920 930 940 950 960 970 980 990 1000 \
  --output /home/nvidia/Desktop/cxr_multi_sensor/yyf/calibration_test_output \


  # 不传 --frames，让脚本自动均匀采样（推荐）
python3 stereo_calib.py \
  --data-dir /projects/calib_data/0323_calibration_v1 \
  --bag /projects/cxr_data/2026-3-20/18-48-10.bag \
  --output /home/nvidia/Desktop/cxr_multi_sensor/yyf/calibration_output

# 如果想手动指定帧，不要超过 100 个
python3 stereo_calib.py \
  --data-dir /projects/calib_data/0323_calibration_v1 \
  --bag /projects/cxr_data/2026-3-20/18-48-10.bag \
  --frames 10 20 30 40 50 60 70 80 90 100 \
  --output /home/nvidia/Desktop/cxr_multi_sensor/yyf/calibration_output




  python3 yyf/export_visible_frames.py \
  -t /projects/cxr_data/20260324_1616 \
  -b /projects/cxr_data/2026-3-24/16-16-32.bag \
  -o /projects/calib_data/0324_visible_export_1

  python3 yyf/export_visible_frames.py \
  -t /projects/cxr_data/20260324_1530 \
  -b /projects/cxr_data/2026-3-24/15-30-31.bag \
  -o /projects/calib_data/0324_visible_export_1 \
  --tianmou-range 200:220 \
  --color-range 100:150
--------------------------------------------------

  python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1949 \
  -b /projects/cxr_data/2026-3-24/19-49-15.bag \
  -o /projects/calib_data/0325_21 \
  --tianmou-range 90:110 \
  --color-range 20:40

python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1949_0 \
  -b /projects/cxr_data/2026-3-24/19-49-56.bag \
  -o /projects/calib_data/0325_22 \
  --tianmou-range 90:110 \
  --color-range 20:40


  python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1950 \
  -b /projects/cxr_data/2026-3-24/19-50-31.bag \
  -o /projects/calib_data/0325_23 \
  --tianmou-range 90:110 \
  --color-range 20:40

    python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1951 \
  -b /projects/cxr_data/2026-3-24/19-51-21.bag \
  -o /projects/calib_data/0325_24 \
  --tianmou-range 90:110 \
  --color-range 20:40

  python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1951_0 \
  -b /projects/cxr_data/2026-3-24/19-51-58.bag \
  -o /projects/calib_data/0325_25 \
  --tianmou-range 90:110 \
  --color-range 20:40

    python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1952 \
  -b /projects/cxr_data/2026-3-24/19-52-35.bag \
  -o /projects/calib_data/0325_26 \
  --tianmou-range 90:110 \
  --color-range 20:40

    python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1953 \
  -b /projects/cxr_data/2026-3-24/19-53-14.bag \
  -o /projects/calib_data/0325_27 \
  --tianmou-range 90:110 \
  --color-range 20:40

   

  /projects/calib_data/0325_calibration/
├── 0325_1_color.png   + 0325_1_tianmou.png  (同一位置的标定对)
├── 0325_2_color.png   + 0325_2_tianmou.png
├── ...
└── 0325_16_color.png + 0325_16_tianmou.png


# 单区间（原有写法）
--tianmou-range 200:400 --color-range 100:150

# 多区间（新增）
--tianmou-range "20:24;30:35;58:65" --color-range "20:24;30:35;58:65"

内参
   python3 export_visible_frames.py \
  -t /projects/cxr_data/20260324_1833 \
  -b /projects/cxr_data/2026-3-24/18-33-11.bag \
  -o /projects/calib_data/0325_neican \
 
cd /home/nvidia/Desktop/cxr_multi_sensor/yyf




    python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/20260324_2054 \
  --bag /projects/cxr_data/2026-3-24/20-54-59.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260324_205537.csv \
  -o output_calibration_0324depth_debug.json

   python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/19700101_0806 \
  --bag /projects/cxr_data/1970-1-1/8-6-50.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_19700101_080719.csv \
  -o output_1970.json



sudo mkdir -p /projects/calib_data/output_1970 && sudo chmod 777 /projects/calib_data/output_1970
  python3 yyf/export_aligned_frames.py \
  -i output_1970.json \
  -t /projects/cxr_data/19700101_0806 \
  -b /projects/cxr_data/1970-1-1/8-6-50.bag \
  -o /projects/calib_data/output_1970

  

    depth_tm/ - 天眸视角的 16 位深度图
color_tm/ - 深度伪彩色可视化
blend/ - 深度与天眸图像叠加图


深度图导出----------------------------------------------------------
# 使用原始 16 位深度（新增的 -r 模式）
python3 yyf/export_aligned_frames.py \
    --index output_triple_aligned.json \
    --tianmou-dir /path/to/tianmou \
    --bag /path/to/record.bag \
    --output /path/to/output \
    --use-raw-depth

 python3 yyf/export_aligned_frames.py \
  -i output_1970.json \
  -t /projects/cxr_data/19700101_0806 \
  -b /projects/cxr_data/1970-1-1/8-6-50.bag \
  -o /projects/calib_data/output_1970 \
  --use-raw-depth
# 或者使用之前的处理后深度（默认行为）
python3 yyf/export_aligned_frames.py \
    --index output_triple_aligned.json \
    --tianmou-dir /path/to/tianmou \
    --bag /path/to/record.bag \
    --output /path/to/output

    sudo mkdir -p /projects/calib_data/0324_calibration_depth_16 && sudo chmod 777 /projects/calib_data/0324_calibration_depth_16
  python3 yyf/export_aligned_frames.py \
  -i output_calibration_0324depth.json \
  -t /projects/cxr_data/20260324_2054 \
  -b /projects/cxr_data/2026-3-24/20-54-59.bag \
  -o /projects/calib_data/0324_calibration_depth_16 \
  --use-raw-depth
  没有最后一行就会是伪彩色深度，加上就是z16位深度图

  


      


    标定新数据：/projects/cxr_data/19700101_0944/
              "/projects/cxr_data/1970-1-1/9-44-59.bag"
              "/projects/cxr_data/temporal_files/tianmou_timestamp_19700101_094632.csv"

      

   python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/19700101_0944 \
  --bag /projects/cxr_data/1970-1-1/9-44-59.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_19700101_094632.csv \
  -o output_1970_0944_cali.json
    sudo mkdir -p /projects/calib_data/output_1970_0944_cali && sudo chmod 777 /projects/calib_data/output_1970_0944_cali
  python3 yyf/export_aligned_frames.py \
  -i output_1970_0944_cali.json \
  -t /projects/cxr_data/19700101_0944 \
  -b /projects/cxr_data/1970-1-1/9-44-59.bag \
  -o /projects/calib_data/output_1970_0944_cali \
  --use-raw-depth

     python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/20260326_2346 \
  --bag /projects/cxr_data/2026-3-26/23-46-59.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260326_234803.csv \
  -o output_0326_2346.json
    sudo mkdir -p /projects/calib_data/output_0326_2346  && sudo chmod 777 /projects/calib_data/output_0326_2346
  python3 yyf/export_aligned_frames.py \
  -i output_0326_2346.json \
  -t /projects/cxr_data/20260326_2346 \
  -b /projects/cxr_data/2026-3-26/23-46-59.bag \
  -o /projects/calib_data/output_0326_2346 \
  --use-raw-depth

python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/19700101_0901\
  --bag /projects/cxr_data/1970-1-1/9-2-6.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_19700101_090255.csv \
  -o output_1970_0901.json
    sudo mkdir -p /projects/calib_data/output_1970_0901 && sudo chmod 777 /projects/calib_data/output_1970_0901
  python3 yyf/export_aligned_frames.py \
  -i output_1970_0901.json \
  -t /projects/cxr_data/19700101_0901 \
  -b /projects/cxr_data/1970-1-1/9-2-6.bag \
  -o /projects/calib_data/output_1970_0901 \
  --use-raw-depth
  
     python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/20260327_1624 \
  --bag /projects/cxr_data/2026-3-27/16-24-48.bag \
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260327_162556.csv \
  -o output_0327_1624.json
    sudo mkdir -p /projects/calib_data/output_0327_1624  && sudo chmod 777 /projects/calib_data/output_0327_1624
  python3 yyf/export_aligned_frames.py \
  -i output_0327_1624.json \
  -t /projects/cxr_data/20260327_1624 \
  -b /projects/cxr_data/2026-3-27/16-24-48.bag \
  -o /projects/calib_data/output_0327_1624 \
  --use-raw-depth

重投影（正确标定版）
python depth_rec.py --base /projects/calib_data/output_1970_0901


  重投影深度图
  最简用法（默认使用 0326 标定参数）
python3 yyf/align_depth_to_tianmou.py /projects/calib_data/output_1970_0901
自定义标定参数
python3 yyf/align_depth_to_tianmou.py /path/to/your/dataset \    --calib-dir /path/to/your/calibration_output
自定义子目录名
python3 yyf/align_depth_to_tianmou.py /path/to/dataset \    --depth-dir depth_raw   \   # 深度图目录名（默认: depth）    --tm-dir    tianmou_raw  \   # TianMou 目录名（默认: tianmou）    --output    compare_new      # 输出目录名（默认: compare）
完整示例
# 处理 0326 数据python3 yyf/align_depth_to_tianmou.py /projects/calib_data/output_0326_2346# 处理 1970 数据python3 yyf/align_depth_to_tianmou.py /projects/calib_data/output_1970# 处理你自己的新数据python3 yyf/align_depth_to_tianmou.py /path/to/your/new_dataset
数据集目录结构要求
your_dataset/├── depth/        # 原始深度图, 480×640, uint16 Z16 mm│   ├── depth_0000.png│   ├── depth_0001.png│   └── ...└── tianmou/     # TianMou 原图, 320×640×3, uint8    ├── tianmou_0000.png    ├── tianmou_0001.png

---------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------
后处理总流程
 post_process.py 导出三路对齐索引文件
 python3 yyf/post_process.py \
  --tianmou-dir /projects/cxr_data/20260414_1917 \
  --bag /projects/cxr_data/2026-4-14/19-18-0.bag\
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_20260414_191825.csv \
  -o output_0414_1917.json

  python3 yyf/post_process.py \
    --tianmou-dir /projects/cxr_data/20260415_18-14-07 \
    --bag /projects/cxr_data/2026-4-15/18-14-15.bag \
    --temporal /projects/cxr_data/20260415_18-14-07/tianmou_timestamp_20260415_181440.csv \
    -o output_0415_1814.json
-----------------------------------------------------------------------------------------------
 export_aligned_frames.py 导出图像（一定使用--use-raw-depth）

sudo mkdir -p /projects/calib_data/output_0415_2  && sudo chmod 777 /projects/calib_data/output_0415_2
  python3 yyf/export_aligned_frames.py \
  -i output_0415_2.json \
  -t /projects/cxr_data/20260415_16-25-06 \
  -b /projects/cxr_data/2026-4-15/16-25-14.bag \
  -o /projects/calib_data/output_0415_2 \
  --use-raw-depth

--------------------------------------------------------------------------------------------------
 depth_rec.py 重投影深度图（自带裁剪和对齐！但是没有保留原始z16）
     python yyf/calib/depth_rec.py \
    --base /projects/calib_data/output_1970_0901 \
    --index output_1970_0901.json


    python yyf/calib/depth_rec.py \
    --base /projects/calib_data/output_1970_0901 \
    --index output_1970_0901.json

python yyf/calib/depth_0408.py \
    --base /projects/calib_data/output_0414_1744 \
    --index output_0414_1744.json

                           
        python3 post_process.py \
  --tianmou-dir /projects/cxr_data/19700101_0847\
  --bag /projects/cxr_data/1970-1-1/8-48-1.bag\
  --temporal /projects/cxr_data/temporal_files/tianmou_timestamp_19700101_084848.csv\
  -o output_1970_0847.json

   sudo mkdir -p /projects/calib_data/output_1970_0847  && sudo chmod 777 /projects/calib_data/output_1970_0847
  python3 export_aligned_frames.py \
  -i output_1970_0847.json \
  -t /projects/cxr_data/19700101_0847 \
  -b /projects/cxr_data/1970-1-1/8-48-1.bag \
  -o /projects/calib_data/output_1970_0847 \
  --use-raw-depth



      python3 check_bag_metadata.py /projects/cxr_data/1970-1-1/10-12-5.bag
