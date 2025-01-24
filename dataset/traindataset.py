import os
import cv2
import lmdb
import numpy as np

def create_lmdb(images_folder, annotations_folder, lmdb_path):
    # 创建 LMDB 数据库
    env = lmdb.open(lmdb_path, map_size=1099511627776, readonly=False)
    
    # 获取图像和标签的文件列表
    image_files = sorted(os.listdir(images_folder))
    annotation_files = sorted(os.listdir(annotations_folder))
    
    # 确保图像和标签数量一致
    assert len(image_files) == len(annotation_files), "Image files and annotation files count mismatch"
    
    with env.begin(write=True) as txn:
        for i, (image_file, annotation_file) in enumerate(zip(image_files, annotation_files)):
            # 读取图像
            image_path = os.path.join(images_folder, image_file)
            image = cv2.imread(image_path)
            
            # 确保图像读取成功
            if image is None:
                print(f"Failed to read image {image_file}. Skipping...")
                continue
            
            # 将图像编码为 JPEG 格式的字节流
            _, image_encoded = cv2.imencode('.jpg', image)
            image_data = image_encoded.tobytes()
            
            # 读取标签文件
            annotation_path = os.path.join(annotations_folder, annotation_file)
            with open(annotation_path, 'r') as f:
                annotations = f.readlines()
            
            # 将标签转为字符串形式，可以选择更改格式（例如 JSON, CSV 等）
            annotation_data = '\n'.join(annotations).encode('utf-8')
            
            # 存储图像字节流和标签
            # 使用图像文件名作为键（可以根据需要自定义）
            key = f"{i:08d}".encode('utf-8')
            
            # 将图像数据和标签数据合并成一个元组或字典
            data = {
                'image': image_data,
                'annotations': annotation_data
            }
            
            # 存储在 LMDB 中
            txn.put(key, str(data).encode('utf-8'))
            
            if i % 100 == 0:  # 每100个条目打印一次
                print(f"Processed {i} images")
    
    print(f"LMDB database created at {lmdb_path}")

# 调用函数
images_folder = './RealCE/train/13mm'  # 图像文件夹路径
annotations_folder = './RealCE/train/trans_annos_13mm'  # 标签文件夹路径
lmdb_path = './RealCE/train/val_lmdb'  # 输出的 LMDB 文件路径

create_lmdb(images_folder, annotations_folder, lmdb_path)
