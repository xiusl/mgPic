import cv2
import numpy as np
from PIL import Image
import os
import torch
from ultralytics import YOLO

def detect_and_mark(image_path):
    """检测图像中的主要对象并用红色矩形标记。"""
    # 加载 YOLOv5 模型
    model = YOLO('yolov5s.pt')
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像文件")
    
    # 使用 YOLOv5 进行检测
    results = model(img)
    
    # 初始化边界框的最大范围
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    
    # 遍历所有检测结果，找到最大的边界范围
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 更新最大范围
            min_x = min(min_x, x1)
            min_y = min(min_y, y1)
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
    
    # 如果检测到了对象，绘制最终的红色矩形
    if min_x != float('inf'):
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    
    # 将OpenCV的BGR格式转换为PIL的RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(img_rgb)
    
    return result

def create_visualization(gray, blurred, blurred2, edges, result_img):
    """Create a visualization of image processing steps.
    
    Args:
        gray: Grayscale image
        blurred: First blurred image
        blurred2: Second blurred image
        edges: Edge detection result
        result_img: Final result image with markings
    """
    # Scale images for better visualization
    scale_percent = 50  # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # Resize all images
    gray_resized = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
    blurred_resized = cv2.resize(blurred, dim, interpolation=cv2.INTER_AREA)
    blurred2_resized = cv2.resize(blurred2, dim, interpolation=cv2.INTER_AREA)
    edges_resized = cv2.resize(edges, dim, interpolation=cv2.INTER_AREA)
    result_resized = cv2.resize(result_img, dim, interpolation=cv2.INTER_AREA)
    
    # Convert edges to 3-channel for proper display
    edges_colored = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    gray_colored = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    blurred_colored = cv2.cvtColor(blurred_resized, cv2.COLOR_GRAY2BGR)
    blurred2_colored = cv2.cvtColor(blurred2_resized, cv2.COLOR_GRAY2BGR)
    
    # Create horizontal stack of images
    vis = np.hstack([gray_colored, blurred_colored, blurred2_colored, edges_colored, result_resized])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_pos = 30
    cv2.putText(vis, 'Gray', (width//2 - 30, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Blurred', (width + width//2 - 50, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Blurred2', (2*width + width//2 - 50, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Edges', (3*width + width//2 - 40, y_pos), font, 1, (0, 255, 0), 2)
    cv2.putText(vis, 'Result', (4*width + width//2 - 40, y_pos), font, 1, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Image Processing Steps", vis)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

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
            if filename.startswith('marked_'):
                continue
            image_path = os.path.join(dir_path, filename)
            try:
                # 处理图像
                result = detect_and_mark(image_path)
                
                # 保存结果
                output_path = os.path.join(dir_path, 'marked_' + filename)
                result.save(output_path)
                print(f"处理后的图像已保存为: {output_path}")
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

if __name__ == '__main__':
    main()