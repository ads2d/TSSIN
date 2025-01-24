import model
import torch

def build_backbone(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    # if torch.cuda.is_available():
    #     model.cuda()  # 移动模型到 CUDA 设备
    backbone = model.backbone.__dict__[cfg['type']](**param)
    backbone = backbone.cuda()
    return backbone
