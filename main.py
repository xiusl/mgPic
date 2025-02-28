import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import torch
from ultralytics import YOLO

def detect(image):
    """检测图像中的主要对象并返回其位置信息。
    
    Args:
        image: 图像
        
    Returns:
        tuple: (PIL Image对象, dict) - 处理后的图像和主要目标的位置信息
        位置信息包含以下字段：
        - x: 目标框左上角的x坐标（相对于原始图像的比例）
        - y: 目标框左上角的y坐标（相对于原始图像的比例）
        - width: 目标框的宽度（相对于原始图像的比例）
        - height: 目标框的高度（相对于原始图像的比例）
        - confidence: 检测置信度
    """
    # 转换为 NumPy 数组
    img = np.array(image)

    # 加载 YOLOv5 模型
    model = YOLO('yolov5su.pt')
    
    # 读取图像
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
            'x': int(x - w/2),  
            'y': int(y - h/2),
            'width': int(w),
            'height': int(h),
            'confidence': float(confidences[max_conf_idx])
        }
    return position_info


def  center_image(image, position_info, mark=False):
    """
    根据位置信息，尽可能少的裁剪边缘生成一张新图片，使识别位置居中
    如果边缘很小，则水平或垂直对称不裁剪
    """
    if mark:
        # 标记识别位置
        image = mark_image(image, position_info)

    x, y, width, height = position_info['x'], position_info['y'], position_info['width'], position_info['height']
    
    # 计算上下左右边距
    left_margin = max(0, int(x))
    top_margin = max(0, int(y))
    right_margin = max(0, int(image.width - (x + width)))
    bottom_margin = max(0, int(image.height - (y + height)))
    print(f'left_margin={left_margin}, top_margin={top_margin}, right_margin={right_margin}, bottom_margin={bottom_margin}')

    x1, y1, width1, height1 = 0, 0, image.width, image.height

    mar = 10
    # 计算新的裁剪区域
    if left_margin > mar and right_margin > mar:
        if left_margin > right_margin:
            x1 = left_margin - right_margin
            width1 = image.width - x1
        else:
            x1 = 0
            width1 = image.width - (right_margin - left_margin)
    # if top_margin > mar and bottom_margin > mar:
    #     if top_margin > bottom_margin:
    #         y1 = top_margin - bottom_margin
    #         height1 = image.height - y1
    #     else:
    #         y1 = 0
    #         height1 = image.height - (bottom_margin - top_margin)

    # 裁剪图像
    return image.crop((x1, y1, x1 + width1, y1 + height1))
    

def mark_image(img, position_info):
    """在图像上标记识别位置, 并返回标记后的图像"""
    x, y, width, height = position_info['x'], position_info['y'], position_info['width'], position_info['height']
    draw = ImageDraw.Draw(img)
    draw.rectangle([max(x, 0), max(y, 0), min(x + width, img.width), min(y + height, img.height)], outline="red", width=2)
    return img


def resize_image(img):
    """调整图像大小，宽度为 750，高度根据宽度等比例缩放, 图片清晰度不变"""
    # 获取图像尺寸
    width, height = img.size

    target_width = 750 - 48
    # 计算新的高度
    new_height = int(height * target_width / width)
    # 调整图像大小
    resized_img = img.resize((target_width, new_height), Image.LANCZOS)
    # 应用轻微锐化滤镜以保持细节清晰
    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(resized_img)
    resized_img = enhancer.enhance(1.2)  # 1.2 是适度的锐化参数
    return resized_img

def crop_margin(img):
    position_info = detect(img)
    x, y, width, height = position_info['x'], position_info['y'], position_info['width'], position_info['height']
    top_margin = max(0, int(y))
    bottom_margin = max(0, int(img.height - (y + height)))
    x1, y1, width1, height1 = 0, 0, img.width, img.height
    v_margin = 40
    if top_margin > v_margin:
        y1 = top_margin - v_margin
        height1 = img.height - (top_margin - v_margin)
    if bottom_margin > v_margin:
        height1 = height1 - (bottom_margin - v_margin)

    return img.crop((x1, y1, x1 + width1, y1 + height1))
        
def full_rect(img):
    """在图片左、右、下、添加24像素的白色边框"""
    width, height = img.size
    # 创建一个新的图像，宽度为原始宽度加上左右边框的宽度，高度为原始高度加上上下边框的高度
    new_img = Image.new('RGB', (width + 48, height + 48), 'white')
    # 将原始图像粘贴到新图像的中心
    new_img.paste(img, (24, 0))
    return new_img

def save_image(img, image_path, prefix):
    """保存图像到目录，重命名为 prefix_xxx.jpg"""
    image_name = prefix + os.path.basename(image_path)
    img.save(os.path.join(os.path.dirname(image_path), image_name))
    return image_name

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

    # 在目录下创建一个 output 文件夹
    output_dir = os.path.join(dir_path, 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历目录中的所有文件
    resize_images = []
    
    for filename in os.listdir(dir_path):
        if filename.lower().endswith(supported_formats):
            if 'marked_' in filename or 'resize_' in filename or 'center_' in filename:
                continue
            if '(1)' in filename:
                continue
            image_path = os.path.join(dir_path, filename)
            try:
                # 打开原始图像
                img = Image.open(image_path)
                
                # 识别图像
                position_info = detect(img)

                # 标记图像
                # img1 = mark_image(img, position_info)
                # save_image(img1, image_path, 'marked_')

                # 居中图像
                img2 = center_image(img, position_info, mark=False)
                # save_image(img2, os.path.join(output_dir, filename),'center_')

                # 调整图像大小
                img3 = resize_image(img2)

                # 识别内容，缩小边距
                img4 = crop_margin(img3)
                img5 = full_rect(img4)
                img_name = save_image(img5, os.path.join(output_dir, filename), 'resize_')

                resize_images.append(img5)
                print(f"处理后的图像已保存为: {img_name}")
                processed_count += 1

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

    # 将处理后的全部图像拼接为一张长图
    if len(resize_images) > 0:
        # 计算拼接后的宽度和高度
        total_height = sum(image.height for image in resize_images)
        max_width = max(image.width for image in resize_images)
        # 创建一个新的空白图像
        result_image = Image.new('RGB', (max_width, total_height))
        # 将每个图像粘贴到新图像中
        y_offset = 0
        for image in resize_images:
            result_image.paste(image, (0, y_offset))
            y_offset += image.height
        # 保存拼接后的图像在 output 文件夹
        result_image.save(os.path.join(output_dir, 'result.jpg'))
        print(f"拼接后的图像已保存为: {os.path.join(output_dir,'result.jpg')}")


if __name__ == '__main__':
    main()