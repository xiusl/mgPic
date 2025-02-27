import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import torch
from ultralytics import YOLO

def detect(image_path):
    """检测图像中的主要对象并返回其位置信息。
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        tuple: (PIL Image对象, dict) - 处理后的图像和主要目标的位置信息
        位置信息包含以下字段：
        - x: 目标框左上角的x坐标（相对于原始图像的比例）
        - y: 目标框左上角的y坐标（相对于原始图像的比例）
        - width: 目标框的宽度（相对于原始图像的比例）
        - height: 目标框的高度（相对于原始图像的比例）
        - confidence: 检测置信度
    """
    # 加载 YOLOv5 模型
    model = YOLO('yolov5su.pt')
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 使用 YOLOv5 进行检测
    results = model(img)
    
    # 获取原始图像尺寸
    height, width = img.shape[:2]
    
    # 初始化位置信息
    position_info = None
    
    # 获取检测结果
    if len(results) > 0 and len(results[0].boxes) > 0:
        # 获取所有检测框
        boxes = results[0].boxes
        # 获取置信度最高的检测框
        confidences = boxes.conf.cpu().numpy()
        max_conf_idx = np.argmax(confidences)
        
        # 获取边界框坐标（归一化后的值）
        box = boxes.xywh[max_conf_idx].cpu().numpy()
        x, y, w, h = box
        print(f'检测到目标框: x={x}, y={y}, w={w}, h={h}')
        # 创建位置信息字典
        position_info = {
            'x': float(x / width),  # 转换为相对坐标
            'y': float(y / height),
            'width': float(w / width),
            'height': float(h / height),
            'confidence': float(confidences[max_conf_idx])
        }
    return position_info


def  make_image(image_path, position_info):
    """根据位置信息，通过裁剪生成一张新图片，让位置居中"""
    # 打开原始图像
    img = Image.open(image_path)
    # 获取图像尺寸
    width, height = img.size
    # 计算目标框的尺寸和位置
    target_width = width * position_info['width']
    target_height = height * position_info['height']
    target_x = width * position_info['x'] - target_width / 2
    target_y = height * position_info['y'] - target_height / 2

    left_margin = target_x
    right_margin = width - (target_x + target_width)
    if left_margin > right_margin:
        target_x = right_margin	- left_margin
    else:
        target_x = 0
    target_width = width - abs(left_margin-right_margin)

    top_margin = target_y
    bottom_margin = height - (target_y + target_height)
    if top_margin > bottom_margin:
        target_y = bottom_margin - top_margin
    else:
        target_y = 0
    target_height = height - abs(top_margin-bottom_margin)
    
    # 裁剪图像
    cropped_img = img.crop((target_x, target_y, target_width, target_height))
    # 保存图像到目录，重名为 center_xxx.jpg
    cropped_img.save(os.path.join(os.path.dirname(image_path), 'center_' + os.path.basename(image_path)))
    return cropped_img


def main():
    import os
    import sys

    # 从命令行获取目录路径，如果未提供则使用当前目录
    dir_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # 处理目录中的每个图像
    processed_count = 0
    error_count = 0
    
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(supported_formats):
            if 'marked_' in filename:
                continue
            image_path = os.path.join(dir_path, filename)
            try:
                # 处理图像
                position_info = detect(image_path)
                
                # 根据 position_info 使用红色边框标记图像，并保存结果
                result = Image.open(image_path)
                if position_info:
                    print(f"位置信息: {position_info}")
                    # 获取图像实际尺寸
                    img_width, img_height = result.size
                    # 将中心坐标转换为左上角坐标
                    x = int(position_info['x'] * img_width - position_info['width'] * img_width/2)
                    y = int(position_info['y']* img_height - position_info['height']* img_height/2)
                    width = int(position_info['width']* img_width)
                    height = int(position_info['height']* img_height)
                    draw = ImageDraw.Draw(result)
                    draw.rectangle([x, y, x + width, y + height], outline="red", width=2)
                    print(f"已标记图像: {filename}")
                else:
                    print(f"未找到位置信息，无法标记图像: {filename}")

                # 保存结果
                output_path = os.path.join(dir_path, 'marked_' + filename)
                result.save(output_path)
                print(f"处理后的图像已保存为: {output_path}")
                processed_count += 1

                # 生成新图片
                make_image(output_path, position_info)
                
            except Exception as e:
                print(f"处理图像 {filename} 时出错: {str(e)}")
                error_count += 1
    
    # 打印处理摘要
    print(f"\n处理完成！")
    print(f"成功处理: {processed_count} 张图像")
    if error_count > 0:
        print(f"处理失败: {error_count} 张图像")
    if processed_count == 0:
        print("在目录中未找到支持的图像格式。")

if __name__ == '__main__':
    main()