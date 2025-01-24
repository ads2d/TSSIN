import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed
import torchvision.models as models
import os

sys.path.append('./')
sys.path.append('../')
from model import PSENet  # 确保你已正确导入PSENet模型
model = dict(
    type='PSENet',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPN',
        in_channels=(256, 512, 1024, 2048),
        out_channels=128
    ),
    detection_head=dict(
        type='PSENet_Head',
        in_channels=1024,
        hidden_dim=256,
        num_classes=4,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.7
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.3
        )
    )
)
data = dict(
    batch_size=2,
    train=dict(
        type='PSENET_IC15',
        split='train',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_num=7,
        min_scale=0.4,
        read_type='cv2'
    ),
    test=dict(
        type='PSENET_IC15',
        split='test',
        short_size=736,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule=(200, 400,),
    epoch=700,
    optimizer='SGD'
)
test_cfg = dict(
    min_score=0.85,
    min_area=16,
    kernel_num=7,
    bbox_type='rect',
    result_path='outputs/submit_ic15.zip'
)

backbone_config = model['backbone']
neck_config = model['neck']
detection_head_config = model['detection_head']

def load_and_transform_image(image_path, input_size):
    """
    加载图片并进行预处理。
    :param image_path: 图片路径。
    :param input_size: 模型输入尺寸，如 (H, W)。
    :return: 预处理后的图片张量。
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 根据你的模型训练设置调整
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 增加批次维度
    return image_tensor


def visualize_psenet_output(psenet_out):
    """
    可视化PSENet的输出特征图。
    :param psenet_out: PSENet模型的输出，形状为(batch_size, num_channels, height, width)
    """
    num_channels = psenet_out.size(1)
    plt.figure(figsize=(20, 10))
    for channel in range(num_channels):
        plt.subplot(1, num_channels, channel + 1)
        feature_map = psenet_out[0, channel].detach().cpu().numpy()
        plt.imshow(feature_map, cmap='gray')
        plt.title(f'Channel {channel}')
        plt.axis('off')
    plt.show()


def main():
    psenet = PSENet(backbone=backbone_config, neck=neck_config, detection_head=detection_head_config)
    model = psenet  # 请根据你的模型加载代码调整这里
    model.eval()  # 设置为评估模式

    # 加载并预处理图片
    image_path = './1714210679829.jpg'  # 设置图片路径
    input_size = (56, 219)  # 根据模型要求设置输入大小
    image_tensor = load_and_transform_image(image_path, input_size)

    # 使用PSENet模型获取输出
    psenet_out = model(image_tensor)

    # 可视化输出
    visualize_psenet_output(psenet_out)
    result_folder = './dataset/image'  # Specify the folder where you want to save the results
    os.makedirs(result_folder, exist_ok=True)  # Create the folder if it doesn't exist
    result_path = os.path.join(result_folder, '7.jpg')  # Define the path for saving the result
    plt.savefig(result_path)  # Save the visualization
    print("OK")


if __name__ == "__main__":
    main()
