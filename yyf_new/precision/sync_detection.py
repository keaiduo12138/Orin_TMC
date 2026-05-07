#!/usr/bin/env python3
"""
精度检测脚本：检测天眸RGB和Realsense深度图的时间对齐精度

假设：
- 图像分辨率：640x320
- 大圆半径：约160像素（占图像1/2）
- 小圆半径：约32像素（约为大圆半径的1/5）
- 旋转速度：匀速，待定数值
"""

import cv2
import numpy as np
import os
import json
from pathlib import Path


def detect_circles(image, is_depth=False):
    """
    检测图像中的圆（大小圆）

    Args:
        image: 输入图像（RGB或深度灰度图）
        is_depth: 是否为深度图（深度图边缘可能模糊）

    Returns:
        circles: 检测到的圆列表，每项为 (cx, cy, r)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if is_depth:
        # 深度图可能需要更强的对比度增强
        gray = cv2.equalizeHist(gray)

    # 边缘检测
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)

    # Hough圆检测
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=300
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # 按半径排序
        circles = circles[circles[:, 2].argsort()[::-1]]
        return circles
    return []


def find_big_small_circles(circles):
    """
    从检测到的圆中区分大圆和小圆

    Args:
        circles: 检测到的圆列表

    Returns:
        (big_circle, small_circle) or (None, None)
    """
    if len(circles) < 2:
        return None, None

    # 假设最大的两个圆是大圆和小圆
    # 大圆半径约为小圆半径的5倍
    sorted_circles = sorted(circles, key=lambda x: x[2], reverse=True)

    big_circle = sorted_circles[0]
    small_circle = None

    # 找到第一个在合理比例范围内的圆作为小圆
    for circle in sorted_circles[1:]:
        # 小圆半径应该约为大圆的1/5-1/3
        ratio = big_circle[2] / circle[2] if circle[2] > 0 else 0
        if 3 < ratio < 8:
            small_circle = circle
            break

    # 如果没找到合适比例的，取第二大的
    if small_circle is None and len(sorted_circles) >= 2:
        small_circle = sorted_circles[1]

    # 验证小圆是否在大圆内部
    if small_circle is not None:
        dx = small_circle[0] - big_circle[0]
        dy = small_circle[1] - big_circle[1]
        dist = np.sqrt(dx**2 + dy**2)
        # 小圆中心到大圆中心的距离应该小于大圆半径
        if dist > big_circle[2]:
            # 小圆不在大圆内，尝试找其他候选
            for circle in sorted_circles[2:]:
                dx = circle[0] - big_circle[0]
                dy = circle[1] - big_circle[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < big_circle[2]:
                    small_circle = circle
                    break

    return big_circle, small_circle


def calculate_angle(cx1, cy1, cx2, cy2):
    """
    计算两点连线与水平轴的角度（度）

    Args:
        cx1, cy1: 第一个点坐标
        cx2, cy2: 第二个点坐标

    Returns:
        角度（度），范围 [-180, 180]
    """
    dx = cx2 - cx1
    dy = cy2 - cy1
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle


def process_image_sequence(image_path, is_depth=False):
    """
    处理一组图像序列，计算每帧的大小圆圆心连线角度

    Args:
        image_path: 图像文件路径
        is_depth: 是否为深度图

    Returns:
        dict: 包含检测结果的字典
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    circles = detect_circles(image, is_depth)
    if len(circles) < 2:
        return None

    big_circle, small_circle = find_big_small_circles(circles)
    if big_circle is None or small_circle is None:
        return None

    angle = calculate_angle(
        big_circle[0], big_circle[1],
        small_circle[0], small_circle[1]
    )

    return {
        'big_circle': big_circle,
        'small_circle': small_circle,
        'angle': angle
    }


def visualize_result(image, result, label, output_path):
    """
    可视化检测结果

    Args:
        image: 原始图像
        result: 检测结果
        label: 标签（tianmou或depth）
        output_path: 输出路径
    """
    if result is None:
        return

    vis_image = image.copy()

    # 画大圆
    cv2.circle(vis_image,
               (result['big_circle'][0], result['big_circle'][1]),
               result['big_circle'][2],
               (0, 255, 0), 2)
    cv2.circle(vis_image,
               (result['big_circle'][0], result['big_circle'][1]),
               3, (0, 255, 0), -1)

    # 画小圆
    cv2.circle(vis_image,
               (result['small_circle'][0], result['small_circle'][1]),
               result['small_circle'][2],
               (0, 0, 255), 2)
    cv2.circle(vis_image,
               (result['small_circle'][0], result['small_circle'][1]),
               3, (0, 0, 255), -1)

    # 画连线和角度
    cv2.line(vis_image,
             (result['big_circle'][0], result['big_circle'][1]),
             (result['small_circle'][0], result['small_circle'][1]),
             (255, 255, 0), 2)

    # 画水平参考线
    h, w = image.shape[:2]
    cv2.line(vis_image, (0, result['big_circle'][1]), (w, result['big_circle'][1]),
             (128, 128, 128), 1)

    # 添加文字
    text = f"{label}: angle={result['angle']:.2f}deg"
    cv2.putText(vis_image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, vis_image)


def process_sequence(tianmou_dir, depth_dir, output_dir, w_assumed=30.0):
    """
    处理整个图像序列，检测时间对齐精度

    Args:
        tianmou_dir: 天眸图像目录
        depth_dir: 深度图像目录
        output_dir: 输出目录
        w_assumed: 假设的旋转角速度（度/秒）

    Returns:
        dict: 分析结果
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取文件列表
    tianmou_files = sorted(Path(tianmou_dir).glob("*.png"))
    tianmou_files += sorted(Path(tianmou_dir).glob("*.jpg"))
    depth_files = sorted(Path(depth_dir).glob("*.png"))
    depth_files += sorted(Path(depth_dir).glob("*.jpg"))

    if len(tianmou_files) == 0 or len(depth_files) == 0:
        print(f"警告: 找不到图像文件")
        print(f"天眸目录: {tianmou_dir}")
        print(f"深度目录: {depth_dir}")
        return None

    print(f"天眸图像数量: {len(tianmou_files)}")
    print(f"深度图像数量: {len(depth_files)}")

    # 处理每帧
    tianmou_angles = []
    depth_angles = []
    tianmou_results = []
    depth_results = []

    for i, (tm_file, depth_file) in enumerate(zip(tianmou_files, depth_files)):
        # 处理天眸
        tm_result = process_image_sequence(str(tm_file), is_depth=False)
        if tm_result:
            tianmou_angles.append(tm_result['angle'])
            tianmou_results.append(tm_result)
        else:
            tianmou_angles.append(None)
            tianmou_results.append(None)

        # 处理深度图
        depth_result = process_image_sequence(str(depth_file), is_depth=True)
        if depth_result:
            depth_angles.append(depth_result['angle'])
            depth_results.append(depth_result)
        else:
            depth_angles.append(None)
            depth_results.append(None)

        # 可视化第一帧
        if i == 0:
            tm_img = cv2.imread(str(tm_file))
            depth_img = cv2.imread(str(depth_file))
            if tm_result:
                visualize_result(tm_img, tm_result, "Tianmou",
                               os.path.join(output_dir, "first_frame_tianmou.png"))
            if depth_result:
                visualize_result(depth_img, depth_result, "Depth",
                               os.path.join(output_dir, "first_frame_depth.png"))

    # 计算角度差值
    tm_diffs = []
    depth_diffs = []

    for i in range(1, len(tianmou_angles)):
        if tianmou_angles[i] is not None and tianmou_angles[i-1] is not None:
            diff = tianmou_angles[i] - tianmou_angles[i-1]
            # 处理角度跨越-180/180的情况
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            tm_diffs.append(diff)
        else:
            tm_diffs.append(None)

        if depth_angles[i] is not None and depth_angles[i-1] is not None:
            diff = depth_angles[i] - depth_angles[i-1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            depth_diffs.append(diff)
        else:
            depth_diffs.append(None)

    # 计算同步误差
    sync_errors = []
    for i in range(len(tm_diffs)):
        if tm_diffs[i] is not None and depth_diffs[i] is not None:
            error = tm_diffs[i] - depth_diffs[i]
            if error > 180:
                error -= 360
            elif error < -180:
                error += 360
            sync_errors.append(error)
        else:
            sync_errors.append(None)

    # 统计分析
    valid_errors = [e for e in sync_errors if e is not None]
    if valid_errors:
        mean_error = np.mean(valid_errors)
        std_error = np.std(valid_errors)
        max_error = np.max(np.abs(valid_errors))
    else:
        mean_error = std_error = max_error = None

    # 输出结果
    print("\n" + "="*60)
    print("精度检测结果")
    print("="*60)

    print(f"\n天眸角度序列 (共{len(tianmou_angles)}帧):")
    for i, angle in enumerate(tianmou_angles[:10]):
        print(f"  帧{i}: {angle:.2f}°" if angle else f"  帧{i}: 检测失败")
    if len(tianmou_angles) > 10:
        print(f"  ... 共{len(tianmou_angles)}帧")

    print(f"\n天眸角度差分 (共{len(tm_diffs)}个):")
    for i, diff in enumerate(tm_diffs[:10]):
        print(f"  帧{i}->{i+1}: {diff:.2f}°" if diff else f"  帧{i}->{i+1}: 缺失")
    if len(tm_diffs) > 10:
        print(f"  ... 共{len(tm_diffs)}个差分")

    print(f"\n深度图角度序列 (共{len(depth_angles)}帧):")
    for i, angle in enumerate(depth_angles[:10]):
        print(f"  帧{i}: {angle:.2f}°" if angle else f"  帧{i}: 检测失败")
    if len(depth_angles) > 10:
        print(f"  ... 共{len(depth_angles)}帧")

    print(f"\n深度图角度差分 (共{len(depth_diffs)}个):")
    for i, diff in enumerate(depth_diffs[:10]):
        print(f"  帧{i}->{i+1}: {diff:.2f}°" if diff else f"  帧{i}->{i+1}: 缺失")
    if len(depth_diffs) > 10:
        print(f"  ... 共{len(depth_diffs)}个差分")

    print(f"\n同步误差统计:")
    if valid_errors:
        print(f"  平均误差: {mean_error:.4f}°")
        print(f"  标准差: {std_error:.4f}°")
        print(f"  最大绝对误差: {max_error:.4f}°")

        print(f"\n同步误差序列 (前10个):")
        for i, err in enumerate(sync_errors[:10]):
            print(f"  帧{i}->{i+1}: {err:.4f}°" if err else f"  帧{i}->{i+1}: 缺失")
    else:
        print("  无法计算同步误差（数据不足）")

    # 保存结果到JSON
    results = {
        'w_assumed': w_assumed,
        'tianmou_angles': tianmou_angles,
        'depth_angles': depth_angles,
        'tm_diffs': tm_diffs,
        'depth_diffs': depth_diffs,
        'sync_errors': sync_errors,
        'statistics': {
            'mean_error': float(mean_error) if mean_error else None,
            'std_error': float(std_error) if std_error else None,
            'max_abs_error': float(max_error) if max_error else None,
        }
    }

    output_json = os.path.join(output_dir, "sync_analysis.json")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_json}")

    return results


def main():
    """主函数"""
    # 配置路径
    base_dir = "/projects/calib_data"
    output_base_dir = "/home/nvidia/Desktop/cxr_multi_sensor/yyf/precision/output"

    # 假设的旋转角速度（度/秒），待实际拍摄后确定
    w_assumed = 30.0  # 可以调整这个值

    # 如果有具体的数据集目录，按顺序处理
    calib_dirs = sorted(Path(base_dir).glob("output_*"))

    if calib_dirs:
        for calib_dir in calib_dirs[:1]:  # 只处理第一个
            output_id = calib_dir.name
            tianmou_dir = calib_dir / "tianmou"
            depth_dir = calib_dir / "depth_tm_0408"
            output_dir = os.path.join(output_base_dir, output_id)

            print(f"\n处理数据集: {output_id}")
            print(f"  天眸目录: {tianmou_dir}")
            print(f"  深度目录: {depth_dir}")

            if tianmou_dir.exists() and depth_dir.exists():
                process_sequence(str(tianmou_dir), str(depth_dir), output_dir, w_assumed)
            else:
                print(f"  警告: 目录不存在，跳过")
    else:
        print(f"在 {base_dir} 中找不到数据目录")
        print("请确保数据目录存在，格式如: /projects/calib_data/output_XXXX/")
        print("  - /projects/calib_data/output_XXXX/tianmou/ (天眸RGB图像)")
        print("  - /projects/calib_data/output_XXXX/depth_tm_0408/ (深度图像)")


if __name__ == "__main__":
    main()
