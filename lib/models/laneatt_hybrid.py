import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

class LightFeatureCut(nn.Module):
    def __init__(self, fmap_h, fmap_w, n_anchors):
        super().__init__()
        self.fmap_h = fmap_h
        self.fmap_w = fmap_w
        self.n_anchors = n_anchors
        
        # 预计算特征图网格
        self.register_buffer('grid_y', torch.linspace(0, 1, fmap_h))
        self.register_buffer('grid_x', torch.linspace(0, 1, fmap_w))
        
    def forward(self, features, anchors):
        """
        Args:
            features: (B, C, H, W) 特征图
            anchors: (N, 2+2+1+S) 锚点
        Returns:
            cut_features: (B, N, C, H, 1) 切割后的特征
        """
        B, C, H, W = features.shape
        device = features.device
        
        # 1. 计算锚点对应的特征图位置
        anchor_starts = anchors[:, 2:4]  # (N, 2)
        anchor_ends = anchors[:, 4:]     # (N, S)
        
        # 2. 生成特征图网格
        grid_y = self.grid_y.to(device).view(1, 1, -1, 1)  # (1, 1, H, 1)
        grid_x = self.grid_x.to(device).view(1, 1, 1, -1)  # (1, 1, 1, W)
        
        # 3. 计算每个锚点的特征图位置
        anchor_positions = torch.zeros(B, self.n_anchors, self.fmap_h, 2, device=device)
        for i in range(self.n_anchors):
            start_y, start_x = anchor_starts[i]
            end_x = anchor_ends[i]
            
            # 计算每个y位置对应的x坐标
            x_coords = start_x + (end_x - start_x) * grid_y.view(-1)
            anchor_positions[:, i, :, 0] = x_coords
            anchor_positions[:, i, :, 1] = grid_y.view(-1)
        
        # 4. 使用双线性插值提取特征
        cut_features = F.grid_sample(
            features,
            anchor_positions.view(B, -1, 1, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).view(B, self.n_anchors, C, self.fmap_h, 1)
        
        return cut_features

class LaneATT_Hybrid(nn.Module):
    def __init__(self,
                 backbone='resnet18',
                 pretrained_backbone=True,
                 S=72,
                 img_w=640,
                 img_h=360,
                 anchor_feat_channels=64,
                 anchors_freq_path=None,
                 topk_anchors=None):
        super().__init__()
        # 基础配置
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = S - 1
        self.n_offsets = S
        self.stride = 32  # ResNet18的stride
        self.fmap_h = img_h // self.stride
        self.fmap_w = img_w // self.stride
        self.anchor_feat_channels = anchor_feat_channels

        # 锚点角度定义
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # 生成锚点
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)
        
        # 根据频率信息过滤锚点
        if anchors_freq_path is not None:
            anchors_mask = torch.load(anchors_freq_path).cpu()
            assert topk_anchors is not None
            ind = torch.argsort(anchors_mask, descending=True)[:topk_anchors]
            self.anchors = self.anchors[ind]
            self.anchors_cut = self.anchors_cut[ind]
            
        self.N = len(self.anchors)  # number of anchors

        # 特征提取器
        self.feature_extractor, backbone_nb_channels, _ = self.get_backbone(backbone, pretrained_backbone)
        self.conv1 = nn.Conv2d(backbone_nb_channels, anchor_feat_channels, kernel_size=1)

        # 混合特征提取
        self.conv_feature = nn.Conv2d(anchor_feat_channels, self.N, kernel_size=1)
        self.light_cut = LightFeatureCut(self.fmap_h, self.fmap_w, self.N)
        
        # 注意力层
        self.attention_layer = nn.Linear(anchor_feat_channels * self.fmap_h, self.N - 1)
        self.initialize_layer(self.attention_layer)
        
        # 特征融合
        self.fusion_conv = nn.Conv1d(anchor_feat_channels * 2, anchor_feat_channels, kernel_size=1)
        
        # 分类和回归头
        self.cls_layer = nn.Conv1d(anchor_feat_channels * self.fmap_h, 2, kernel_size=1)
        self.reg_layer = nn.Conv1d(anchor_feat_channels * self.fmap_h, self.n_offsets + 1, kernel_size=1)

        # 锚点偏移
        self.register_buffer("anchor_xy", self.anchors[:, 2:4].unsqueeze(0))
        self.register_buffer("anchor_template", self.anchors[:, 4:].unsqueeze(0))

    @staticmethod
    def initialize_layer(layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(layer.weight, mean=0., std=0.001)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def get_backbone(self, backbone, pretrained):
        if backbone == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            fmap_c = 512
            stride = 32
        else:
            raise NotImplementedError(f'Backbone {backbone} not implemented')
        
        return nn.Sequential(*list(backbone.children())[:-2]), fmap_c, stride

    def generate_anchors(self, lateral_n, bottom_n):
        left_anchors, left_cut = self.generate_side_anchors(self.left_angles, x=0., nb_origins=lateral_n)
        right_anchors, right_cut = self.generate_side_anchors(self.right_angles, x=1., nb_origins=lateral_n)
        bottom_anchors, bottom_cut = self.generate_side_anchors(self.bottom_angles, y=1., nb_origins=bottom_n)

        return torch.cat([left_anchors, bottom_anchors, right_anchors]), torch.cat([left_cut, bottom_cut, right_cut])

    def generate_side_anchors(self, angles, nb_origins, x=None, y=None):
        if x is None and y is not None:
            starts = [(x, y) for x in np.linspace(1., 0., num=nb_origins)]
        elif x is not None and y is None:
            starts = [(x, y) for y in np.linspace(1., 0., num=nb_origins)]
        else:
            raise Exception('Please define exactly one of `x` or `y` (not neither nor both)')

        n_anchors = nb_origins * len(angles)
        anchors = torch.zeros((n_anchors, 2 + 2 + 1 + self.n_offsets))
        anchors_cut = torch.zeros((n_anchors, 2 + 2 + 1 + self.fmap_h))

        for i, start in enumerate(starts):
            for j, angle in enumerate(angles):
                k = i * len(angles) + j
                anchors[k] = self.generate_anchor(start, angle)
                anchors_cut[k] = self.generate_anchor(start, angle, cut=True)

        return anchors, anchors_cut

    def generate_anchor(self, start, angle, cut=False):
        if cut:
            anchor_ys = torch.linspace(1, 0, steps=self.fmap_h, dtype=torch.float32)
            anchor = torch.zeros(2 + 2 + 1 + self.fmap_h)
        else:
            anchor_ys = torch.linspace(1, 0, steps=self.n_offsets, dtype=torch.float32)
            anchor = torch.zeros(2 + 2 + 1 + self.n_offsets)
        
        angle = angle * math.pi / 180.  # degrees to radians
        start_x, start_y = start
        anchor[2] = 1 - start_y
        anchor[3] = start_x
        anchor[5:] = (start_x + (1 - anchor_ys - 1 + start_y) / math.tan(angle)) * self.img_w

        return anchor

    def forward(self, x):
        # 1. 特征提取
        x = self.feature_extractor(x)     # (B, 512, 11, 20)
        x = self.conv1(x)                 # (B, 64, 11, 20)
        
        # 2. 卷积特征提取
        conv_feat = self.conv_feature(x)  # (B, N, 11, 20)
        
        # 3. 轻量级特征切割
        cut_feat = self.light_cut(x, self.anchors)  # (B, N, 64, 11, 1)
        cut_feat = cut_feat.squeeze(-1)   # (B, N, 64, 11)
        
        # 4. 特征融合
        B, N, C, H = cut_feat.shape
        conv_feat = conv_feat.view(B, N, -1)  # (B, N, 220)
        cut_feat = cut_feat.view(B, N, -1)    # (B, N, 704)
        
        # 5. 特征拼接和融合
        fused_feat = torch.cat([conv_feat, cut_feat], dim=2)  # (B, N, 924)
        fused_feat = fused_feat.transpose(1, 2)  # (B, 924, N)
        fused_feat = self.fusion_conv(fused_feat)  # (B, 64, N)
        
        # 6. 注意力机制
        # 计算注意力分数
        attention_features = fused_feat.transpose(1, 2)  # (B, N, 64)
        attention_features = attention_features.reshape(-1, self.anchor_feat_channels * self.fmap_h)
        scores = self.attention_layer(attention_features)  # (B*N, N-1)
        
        # 应用softmax获取注意力权重
        attention = F.softmax(scores, dim=1).reshape(B, N, -1)  # (B, N, N-1)
        
        # 构建注意力矩阵
        attention_matrix = torch.eye(N, device=x.device).repeat(B, 1, 1)  # (B, N, N)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        
        # 应用注意力
        attention_features = torch.bmm(
            torch.transpose(attention_features.reshape(B, N, -1), 1, 2),
            torch.transpose(attention_matrix, 1, 2)
        ).transpose(1, 2)  # (B, N, 64)
        
        # 7. 分类和回归
        cls_logits = self.cls_layer(attention_features.transpose(1, 2)).transpose(1, 2)  # (B, N, 2)
        reg = self.reg_layer(attention_features.transpose(1, 2)).transpose(1, 2)         # (B, N, 73)
        
        # 8. 生成最终输出
        cls_scores = F.softmax(cls_logits, dim=2)
        offsets = self.anchor_template + reg
        
        proposals = torch.cat([cls_scores, self.anchor_xy.expand(B, -1, -1), offsets], dim=2)
        return proposals 