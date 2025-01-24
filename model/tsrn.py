import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed
import torchvision.models as models
from timm.models.layers import trunc_normal_

sys.path.append('./')
sys.path.append('../')
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .model_transformer import FeatureEnhancer, ReasoningTransformer, FeatureEnhancerW2V
#from .pesnet import PSENet
from model import PSENet

# from .neck import build_neck

#from .psenet import PSENet


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


class TSRN(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2*hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2*hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2*hidden_units, 2*hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2*hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2*hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2*hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height//scale_factor, width//scale_factor]
        tps_outputsize = [height//scale_factor, width//scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        # print("block_keys:", block.keys())
        # print("output:", output.shape)
        return output


class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=(1, 0), bias=False)
        self.bn4 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        return x

#############################################
class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelReducer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

        
        
###################################################################################      
class SalientFeatureExtractor(nn.Module):
    def __init__(self):
        super(SalientFeatureExtractor, self).__init__()
        # 初始化第一个卷积层，直接从64个通道增加到96个通道
        self.conv1 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        
        # 后续层继续处理，但保持通道数为96，不改变空间维度
        self.conv2 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(96)
        # 可以根据需要增加更多的卷积层
        self.conv3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  # 第一层卷积，通道数变为96
        out = self.relu(self.bn2(self.conv2(out)))  # 第二层卷积，通道数保持为96
        out = self.relu(self.bn3(self.conv3(out)))  # 第三层卷积，通道数保持为96
        # 经过上述处理后，输出形状为[48, 96, 16, 64]
        return out



class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2

class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1,
                      bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels)
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction,
                                        norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)

        return merge
################################################

class TSRN_TL(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300,
                 text_emb=37, #26+26+1
                 out_text_channels=32):
        super(TSRN_TL, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.feature_fusion = FeatureFusionModule(dim=4)
        self.denoise_module = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),  # Assuming psenet_out has 4 channels
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.conv = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, padding=0)
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlockTL(2 * hidden_units, out_text_channels))

        # self.w2v_proj = ImFeat2WordVec(2 * hidden_units, word_vec_d)
        # self.semantic_R = ReasoningTransformer(2 * hidden_units)


        self.feature_enhancer = None #FeatureEnhancerW2V(
                                #        vec_d=300,
                                #        feature_size=2 * hidden_units,
                                #        head_num=4,
                                #        dropout=True)

        # From [1, 1] -> [16, 16]
        self.infoGen = InfoGen(text_emb, out_text_channels)
        self.emb_cls = text_emb

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none',
                input_size=self.tps_inputsize)
                
         #self.saliency_feature_extractor = SaliencyFeatureExtractionModule(in_planes, 64)
         ###########################

    def forward(self, x, text_emb=None):
        #saliency_features = self.saliency_feature_extractor(x)
        #################
        # embed()

        # print("self.tps_inputsize:", self.tps_inputsize)
        # print("x.shape is ",x.shape)    # (48.4,16,64)
        x_PSE = x[:, :3, :, :]
        psenet = PSENet(backbone=backbone_config, neck=neck_config, detection_head=detection_head_config)
        #print("psenet shape is ",psenet.shape)
        psenet_out = psenet.forward(x_PSE)
        #psenet_out = psenet_out[:, -4:, :, :]
        psenet_out = F.interpolate(psenet_out, size=(16, 64), mode='bilinear', align_corners=True)
        psenet_out = self.denoise_module(psenet_out)
        #print("psenet_out is ",psenet_out.shape)
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        # all_pred_vecs = []

        if text_emb is None:
            N, C, H, W = x.shape
            text_emb = torch.zeros((N, self.emb_cls, 1, 26))

        text_emb = text_emb.cuda()
        infoGen =self.infoGen.cuda()
        spatial_t_emb = infoGen(text_emb)
        spatial_t_emb = F.interpolate(spatial_t_emb, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        #x = self.feature_fusion(x, psenet_out)
        x = x + psenet_out

        #print("x", x.shape, spatial_t_emb.shape)
        #[48,4,16,64] [48,32,16,64]
        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in [2, 3, 4, 5, 6]:
                # pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                # all_pred_vecs.append(pred_word_vecs)
                # if not self.training:
                #     word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], spatial_t_emb)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])
        output = output.cuda()


        return output


class TSRN_C2F(nn.Module):
    def __init__(self, scale_factor=2, width=128, height=32, STN=False, srb_nums=5, mask=True, hidden_units=32):
        super(TSRN_C2F, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
            # nn.ReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), RecurrentResidualBlock(2 * hidden_units))

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        self.coarse_proj = nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4)

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units + in_planes, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units + in_planes, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        for i in range(self.srb_nums + 1):
            block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        proj_coarse = self.coarse_proj(block[str(self.srb_nums + 2)])
        # block[str(srb_nums + 2)] =

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            (torch.cat([block['1'] + block[str(self.srb_nums + 2)], proj_coarse], axis=1))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        # print("block_keys:", block.keys())

        return output, proj_coarse


class SEM_TSRN(nn.Module):
    def __init__(self,
                 scale_factor=2,
                 width=128,
                 height=32,
                 STN=False,
                 srb_nums=5,
                 mask=True,
                 hidden_units=32,
                 word_vec_d=300):
        super(SEM_TSRN, self).__init__()
        in_planes = 3
        if mask:
            in_planes = 4
        assert math.log(scale_factor, 2) % 1 == 0
        upsample_block_num = int(math.log(scale_factor, 2))
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.srb_nums = srb_nums
        for i in range(srb_nums):
            setattr(self, 'block%d' % (i + 2), ReasoningResidualBlock(2 * hidden_units))

        self.w2v_proj = ImFeat2WordVec(2 * hidden_units, word_vec_d)
        # self.semantic_R = ReasoningTransformer(2 * hidden_units)

        self.feature_enhancer = None #FeatureEnhancerW2V(
                                #        vec_d=300,
                                #        feature_size=2 * hidden_units,
                                #        head_num=4,
                                #        dropout=True)

        setattr(self, 'block%d' % (srb_nums + 2),
                nn.Sequential(
                    nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
                    nn.BatchNorm2d(2 * hidden_units)
                ))

        # self.non_local = NonLocalBlock2D(64, 64)
        block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
        block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
        setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
        self.tps_inputsize = [height // scale_factor, width // scale_factor]
        tps_outputsize = [height // scale_factor, width // scale_factor]
        num_control_points = 20
        tps_margins = [0.05, 0.05]
        self.stn = STN
        if self.stn:
            self.tps = TPSSpatialTransformer(
                output_image_size=tuple(tps_outputsize),
                num_control_points=num_control_points,
                margins=tuple(tps_margins))

            self.stn_head = STNHead(
                in_planes=in_planes,
                num_ctrlpoints=num_control_points,
                activation='none')

    def forward(self, x, word_vecs=None):
        # embed()
        if self.stn and self.training:
            # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            _, ctrl_points_x = self.stn_head(x)
            x, _ = self.tps(x, ctrl_points_x)
        block = {'1': self.block1(x)}

        all_pred_vecs = []

        # Reasoning block: [2, 3, 4, 5, 6]
        for i in range(self.srb_nums + 1):
            if i + 2 in [2, 3, 4, 5, 6]:
                pred_word_vecs = self.w2v_proj(block[str(i + 1)])
                all_pred_vecs.append(pred_word_vecs)
                if not self.training:
                    word_vecs = pred_word_vecs
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)], self.feature_enhancer, word_vecs)
            else:
                block[str(i + 2)] = getattr(self, 'block%d' % (i + 2))(block[str(i + 1)])

        block[str(self.srb_nums + 3)] = getattr(self, 'block%d' % (self.srb_nums + 3)) \
            ((block['1'] + block[str(self.srb_nums + 2)]))
        output = torch.tanh(block[str(self.srb_nums + 3)])

        return output, all_pred_vecs


class RecurrentResidualBlock(nn.Module):
    def __init__(self, channels):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.gru1(residual.transpose(-1, -2)).transpose(-1, -2)
        # residual = self.non_local(residual)

        return self.gru2(x + residual)
####################################
class FeatureRefinementModule(nn.Module):
    def __init__(self, channels=96, activation_fn=nn.ReLU):
        super(FeatureRefinementModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation = activation_fn(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)
        return out

#######################################
      
class RecurrentResidualBlockTL(nn.Module):
    def __init__(self, channels, text_channels):
        super(RecurrentResidualBlockTL, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.gru1 = GruBlock(channels + text_channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gru2 = GruBlock(channels, channels)
        #self.change_channels = nn.Conv2d(in_channels=100, out_channels=96, kernel_size=1, stride=1, padding=0)
        ######添加Feature Refinement Module#
        self.feature_refine = FeatureRefinementModule(96)
        ##############
        

    def forward(self, x, text_emb):
        #x_PSE = x[:, :3, :, :]
        #psenet = PSENet(backbone=backbone_config, neck=neck_config, detection_head=detection_head_config)
        #print("psenet shape is ",psenet.shape)
        #psenet_out = psenet.forward(x_PSE)
        #psenet_out = psenet_out[:, -96:, :, :]
        #psenet_out = F.interpolate(psenet_out, size=(16, 64), mode='bilinear', align_corners=True)
        #print("psenet_out is ",psenet_out.shape)
        ######################################
        #salient_feature = self.salient_feature_extractor(x)
        ##################
        #print("salient_feature shape  is ",salient_feature.shape)
        #x = x.cuda()
        #conv1 = self.conv1.cuda()
        residual = self.conv1(x)
        #bn1 = self.bn1.cuda()
        residual = self.bn1(residual)
        # prelu = slef.prelu.cuda()
        residual = self.prelu(residual)
        #conv2 = self.conv2.cuda()
        residual = self.conv2(x)
        # residual = self.conv2(residual)
        #bn2 = self.bn2.cuda()
        residual = self.bn2(residual)
        #text_emb = text_emb.cuda()
        cat_feature = torch.cat([residual, text_emb], 1)
        #cat_feature = cat_feature.cuda()
        #print("cat_feature is on ",cat_feature.device)
        #print("psenet_out is on ", psenet_out.device)
        #cat_feature = cat_feature + psenet_out

        #########################
        #cat_feature2 = torch.cat([cat_feature, salient_feature], 1)
        #cat_feature = self.change_channels(cat_feature)
        ########################################
        
        #print(" later cat_feature2 shape:",cat_feature2.shape)
        #########在融合特征上应用特征细化模块##
        #cat_feature = self.feature_refine(cat_feature)
        ####################
        #grul = self.gru1.cuda()
        residual = self.gru1(cat_feature.transpose(-1, -2)).transpose(-1, -2)
        #print("residual shape:",residual.shape)
        #gru2 = self.gru2.cuda()吧

        # t = (x + residual).cuda()
        #print("forward over ")
        return self.gru2(x + residual)



class ReasoningResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ReasoningResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.gru1 = GruBlock(channels, channels)
        # self.prelu = nn.ReLU()
        self.prelu = mish()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        # self.gru2 = GruBlock(channels, channels)
        self.feature_enhancer = FeatureEnhancerW2V(
                                        vec_d=300,
                                        feature_size=channels,
                                        head_num=4,
                                        dropout=0.1)

    def forward(self, x, feature_enhancer, wordvec):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        size = residual.shape
        # residual: [N, C, H, W];
        # wordvec: [N, C_vec]
        residual = residual.view(size[0], size[1], -1)
        residual = self.feature_enhancer(residual, wordvec)
        residual = residual.resize(size[0], size[1], size[2], size[3])

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)

        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        # self.prelu = nn.ReLU()
        self.prelu = mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class mish(nn.Module):
    def __init__(self, ):
        super(mish, self).__init__()
        self.activated = True

    def forward(self, x):
        if self.activated:
            x = x * (torch.tanh(F.softplus(x)))
        return x


class GruBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GruBlock, self).__init__()
        assert out_channels % 2 == 0
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.gru = nn.GRU(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        b = x.size()
        x = x.view(b[0] * b[1], b[2], b[3])
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        # x = self.gru(x)[0]
        x = x.view(b[0], b[1], b[2], b[3])
        x = x.permute(0, 3, 1, 2)
        return x


class ImFeat2WordVec(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImFeat2WordVec, self).__init__()
        self.vec_d = out_channels
        self.vec_proj = nn.Linear(in_channels, self.vec_d)

    def forward(self, x):

        b, c, h, w = x.size()
        result = x.view(b, c, h * w)
        result = torch.mean(result, 2)
        pred_vec = self.vec_proj(result)

        return pred_vec


if __name__ == '__main__':
    # net = NonLocalBlock2D(in_channels=32)
    img = torch.zeros(7, 3, 16, 64)
    embed()
