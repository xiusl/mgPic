from PIL import Image
import os

def stitch_images(image_paths):
    """将多个图像垂直拼接成一个长图像。"""
    # 打开所有图像并确保它们都是相同的模式
    images = [Image.open(path) for path in image_paths]
    
    # 将所有图像转换为RGB模式（如果不是RGB模式）
    images = [img.convert('RGB') if img.mode != 'RGB' else img for img in images]
    
    # 获取所有图像中的最大宽度
    max_width = max(img.width for img in images)
    
    # 计算总高度
    total_height = sum(img.height for img in images)
    
    # 创建一个新的空白图像
    stitched_image = Image.new('RGB', (max_width, total_height))
    
    # 粘贴每个图像
    y_offset = 0
    for img in images:
        # 如果图像宽度小于最大宽度，将其居中
        x_offset = (max_width - img.width) // 2
        stitched_image.paste(img, (x_offset, y_offset))
        y_offset += img.height
        
    return stitched_image

def split_image(image, target_width, target_height):
    """根据目标尺寸将图像分割成多个部分。"""
    width, height = image.size
    pieces = []
    
    # 计算行数和列数
    num_cols = (width + target_width - 1) // target_width
    num_rows = (height + target_height - 1) // target_height
    
    for row in range(num_rows):
        for col in range(num_cols):
            # 计算裁剪坐标
            left = col * target_width
            upper = row * target_height
            right = min((col + 1) * target_width, width)
            lower = min((row + 1) * target_height, height)
            
            # 裁剪图像片段
            piece = image.crop((left, upper, right, lower))
            pieces.append(piece)
    
    return pieces

def main():
    # 示例用法
    # 获取要拼接的图像路径列表
    image_paths = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            image_paths.append(file)
    
    if not image_paths:
        print("当前目录中未找到图像文件！")
        return
    
    # 拼接图像
    print("正在拼接图像...")
    stitched_image = stitch_images(image_paths)
    
    # 保存拼接后的图像
    stitched_image.save('stitched_image.jpg')
    print("拼接后的图像已保存为 'stitched_image.jpg'")
    
    # 获取分割的目标尺寸
    try:
        target_width = int(input("请输入分割的目标宽度: "))
        target_height = int(input("请输入分割的目标高度: "))
    except ValueError:
        print("请输入有效的宽度和高度数值！")
        return
    
    # 分割拼接后的图像
    print("正在分割图像...")
    pieces = split_image(stitched_image, target_width, target_height)
    
    # 保存分割后的图像片段
    os.makedirs('split_pieces', exist_ok=True)
    for i, piece in enumerate(pieces):
        piece.save(f'split_pieces/piece_{i+1}.jpg')
    
    print(f"已分割成 {len(pieces)} 个片段。保存在 'split_pieces' 目录中。")

if __name__ == '__main__':
    main()