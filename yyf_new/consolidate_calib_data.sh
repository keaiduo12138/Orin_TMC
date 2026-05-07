#!/bin/bash

# 整合0325标定数据
# 从每个0325_X目录提取第一张color和tianmou图片，整合到0325_calibration目录

OUTPUT_DIR="/projects/calib_data/0325_calibration"
SOURCE_BASE="/projects/calib_data"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "整合0325标定数据"
echo "输出目录: $OUTPUT_DIR"
echo "=============================================="

# 遍历所有0325_1到0325_27目录
for i in $(seq 1 27); do
    DIR_NAME=$(printf "0325_%d" $i)
    SRC_DIR="${SOURCE_BASE}/${DIR_NAME}"

    if [ ! -d "$SRC_DIR" ]; then
        echo "[跳过] $DIR_NAME 目录不存在"
        continue
    fi

    # 获取第一张color图片
    COLOR_FILE=$(ls "$SRC_DIR/color/" 2>/dev/null | head -1)
    TIANMOU_FILE=$(ls "$SRC_DIR/tianmou/" 2>/dev/null | head -1)

    if [ -z "$COLOR_FILE" ] || [ -z "$TIANMOU_FILE" ]; then
        echo "[跳过] $DIR_NAME 缺少文件"
        continue
    fi

    # 复制并重命名
    cp "$SRC_DIR/color/$COLOR_FILE" "$OUTPUT_DIR/${DIR_NAME}_color.png"
    cp "$SRC_DIR/tianmou/$TIANMOU_FILE" "$OUTPUT_DIR/${DIR_NAME}_tianmou.png"

    echo "[完成] $DIR_NAME: ${DIR_NAME}_color.png + ${DIR_NAME}_tianmou.png"
done

echo ""
echo "=============================================="
echo "整合完成！文件列表："
ls -la "$OUTPUT_DIR/"
echo "=============================================="
