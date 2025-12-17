#!/bin/sh

# 读取 config.yaml 中的 file_without_suffix 路径
CONFIG_FILE="config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# 提取路径，去除引号和空格
BAG_PATH=$(grep "file_without_suffix:" "$CONFIG_FILE" | awk -F': ' '{print $2}' | tr -d '"' | tr -d "'")

# 展开波浪号 ~
eval BAG_PATH="$BAG_PATH.bag"

echo "Attempting to repair bag file: $BAG_PATH"

if [ -f "$BAG_PATH" ]; then
    rosbag reindex "$BAG_PATH"
else
    echo "Error: Bag file not found: $BAG_PATH"
fi
