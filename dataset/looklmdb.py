import lmdb
import cv2
import numpy as np
import os

def visualize_lmdb(lmdb_path, save_path=None):
    # 打开LMDB环境
    env = lmdb.open(lmdb_path, readonly=True)
    
    # 获取所有的键
    with env.begin() as txn:
        cursor = txn.cursor()
        keys = [key.decode() for key, _ in cursor]  # 解码键
    
    print(f"Total keys in the LMDB: {len(keys)}")
    
    # 假设LMDB存储的是图像和标签数据
    images = []
    labels = []
    
    # 遍历所有的键值对，读取数据
    with env.begin() as txn:
        for key in keys:
            value = txn.get(key.encode())
            
            # 假设存储的是图像和标签
            # 这里我们假设value是字节流，存储的是图像数据（如jpg/PNG格式）
            if value:
                # 将字节数据转为图像
                image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                # 检查图像是否成功读取
                if image is not None:
                    images.append(image)
                    labels.append(key)  # 使用key作为标签，可以替换为实际的标签
                else:
                    print(f"Failed to decode image for key: {key}")
    
    print(f"Number of images successfully read: {len(images)}")

    # 如果指定了保存路径，则保存每个图像
    if len(images) > 0:
        if save_path:
            os.makedirs(save_path, exist_ok=True)  # 如果路径不存在，则创建
            
            # 保存每张图像为独立文件
            for i, image in enumerate(images):
                file_path = os.path.join(save_path, f"image_{i+1}_{labels[i]}.png")
                cv2.imwrite(file_path, image)
                print(f"Saved image: {file_path}")
    else:
        print("No images to save")

# 调用函数来可视化LMDB中的数据，并保存每张图像为独立文件
visualize_lmdb('./textzoom/test/hard', save_path='./saved_images')
