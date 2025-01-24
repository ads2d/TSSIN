import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck
from .head import build_head
#from .utils import Conv_BN_ReLU

from .utils import Conv_BN_ReLU

class PSENet(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 detection_head):
        super(PSENet, self).__init__()
        self.backbone = build_backbone(backbone)
        self.fpn = build_neck(neck)

        self.det_head = build_head(detection_head)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=4, kernel_size=1, padding=0)

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_texts=None,
                gt_kernels=None,
                training_masks=None,
                img_metas=None,
                cfg=None):
        #print("come in PSENET")
        outputs = dict()



        # backbone
        imgs = imgs.cuda()
        backbone = self.backbone.cuda()
        f = backbone(imgs)
        # print("")


        # FPN
        fpn = self.fpn.cuda()
        f1, f2, f3, f4, = fpn(f[0], f[1], f[2], f[3])

        f = torch.cat((f1, f2, f3, f4), 1)

        det_head = self.det_head.cuda()
        det_out = det_head(f)
        psenet_feature = det_out


        return psenet_feature



