import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np
from lib.lane import Lane
from lib.focal_loss import FocalLoss
from nms import nms
from .matching import match_proposals_with_targets
import time

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
        if self.training:
            torch.cuda.synchronize()
            start_time = time.time()
        
        B, C, H, W = features.shape
        device = features.device
        
        # 1. 计算锚点对应的特征图位置
        anchor_starts = anchors[:, 2:4].to(device)  # (N, 2)
        anchor_ends = anchors[:, 4:].to(device)     # (N, S)
        
        # 2. 生成特征图网格
        grid_y = self.grid_y.to(device).view(1, 1, -1, 1)  # (1, 1, H, 1)
        
        # 3. 批量处理所有锚点
        # 扩展维度以支持广播
        start_x = anchor_starts[:, 1].view(1, -1, 1, 1)  # (1, N, 1, 1)
        start_y = anchor_starts[:, 0].view(1, -1, 1, 1)  # (1, N, 1, 1)
        
        # 修改：正确处理维度进行线性插值
        # 将 anchor_ends 重塑为 3D tensor (N, 1, S)
        anchor_ends_3d = anchor_ends.view(self.n_anchors, 1, -1)
        
        # 对每个锚点进行线性插值
        end_x = F.interpolate(
            anchor_ends_3d,  # (N, 1, S)
            size=self.fmap_h,
            mode='linear',
            align_corners=True
        )  # (N, 1, H)
        
        # 重塑回需要的维度
        end_x = end_x.view(1, self.n_anchors, 1, self.fmap_h)  # (1, N, 1, H)
        
        # 计算所有锚点的 x 坐标
        x_coords = start_x + (end_x - start_x) * grid_y  # (1, N, 1, H)
        
        # 构建完整的 anchor_positions
        anchor_positions = torch.zeros(B, self.n_anchors, self.fmap_h, 2, device=device)
        
        # 修复：正确处理维度扩展
        x_coords = x_coords.squeeze(-1)  # (1, N, H)
        x_coords = x_coords.expand(B, -1, -1)  # (B, N, H)
        anchor_positions[..., 0] = x_coords
        
        grid_y = grid_y.squeeze(-1)  # (1, 1, H)
        grid_y = grid_y.expand(B, self.n_anchors, -1)  # (B, N, H)
        anchor_positions[..., 1] = grid_y
        
        # 4. 使用双线性插值提取特征
        cut_features = F.grid_sample(
            features,
            anchor_positions.view(B, -1, 1, 2),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        ).view(B, self.n_anchors, C, self.fmap_h, 1)
        
        if self.training:
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"LightFeatureCut forward time: {(end_time - start_time)*1000:.2f}ms")
        
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
        
        # 特征处理层
        self.conv1 = nn.Conv2d(backbone_nb_channels, anchor_feat_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(anchor_feat_channels, anchor_feat_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(anchor_feat_channels, self.N, kernel_size=1)
        
        # 注意力层
        self.attention_layer = nn.Linear(anchor_feat_channels * self.fmap_h, self.N - 1)
        
        # 分类和回归头
        self.cls_layer = nn.Conv1d(anchor_feat_channels * self.fmap_h, 2, kernel_size=1)
        self.reg_layer = nn.Conv1d(anchor_feat_channels * self.fmap_h, self.n_offsets + 1, kernel_size=1)

        # 锚点偏移
        self.register_buffer("anchor_xy", self.anchors[:, 2:4].unsqueeze(0))
        self.register_buffer("anchor_template", self.anchors[:, 4:].unsqueeze(0))
        
        # 初始化层
        self.initialize_layer(self.conv1)
        self.initialize_layer(self.conv2)
        self.initialize_layer(self.conv3)
        self.initialize_layer(self.attention_layer)
        self.initialize_layer(self.cls_layer)
        self.initialize_layer(self.reg_layer)

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

    def loss(self, proposals_list, targets, cls_loss_weight=10):
        focal_loss = FocalLoss(alpha=0.25, gamma=2.)
        smooth_l1_loss = nn.SmoothL1Loss()
        cls_loss = 0
        reg_loss = 0
        valid_imgs = len(targets)
        total_positives = 0
        
        for proposals, target in zip(proposals_list, targets):
            # 过滤不存在的车道线
            target = target[target[:, 1] == 1]
            if len(target) == 0:
                # 如果没有目标，所有提议都应该是负样本
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue
                
            # 计算正负样本匹配
            with torch.no_grad():
                positives_mask, invalid_offsets_mask, negatives_mask, target_positives_indices = match_proposals_with_targets(
                    self, self.anchors, target)

            positives = proposals[positives_mask]
            num_positives = len(positives)
            total_positives += num_positives
            negatives = proposals[negatives_mask]
            num_negatives = len(negatives)

            # 处理没有正样本的情况
            if num_positives == 0:
                cls_target = proposals.new_zeros(len(proposals)).long()
                cls_pred = proposals[:, :2]
                cls_loss += focal_loss(cls_pred, cls_target).sum()
                continue

            # 获取分类目标
            all_proposals = torch.cat([positives, negatives], 0)
            cls_target = proposals.new_zeros(num_positives + num_negatives).long()
            cls_target[:num_positives] = 1.
            cls_pred = all_proposals[:, :2]

            # 回归目标
            reg_pred = positives[:, 4:]
            with torch.no_grad():
                target = target[target_positives_indices]
                positive_starts = (positives[:, 2] * self.n_strips).round().long()
                target_starts = (target[:, 2] * self.n_strips).round().long()
                target[:, 4] -= positive_starts - target_starts
                all_indices = torch.arange(num_positives, dtype=torch.long)
                ends = (positive_starts + target[:, 4] - 1).round().long()
                invalid_offsets_mask = torch.zeros((num_positives, 1 + self.n_offsets + 1),
                                                   dtype=torch.int, device=proposals.device)
                invalid_offsets_mask[all_indices, 1 + positive_starts] = 1
                invalid_offsets_mask[all_indices, 1 + ends + 1] -= 1
                invalid_offsets_mask = invalid_offsets_mask.cumsum(dim=1) == 0
                invalid_offsets_mask = invalid_offsets_mask[:, :-1]
                invalid_offsets_mask[:, 0] = False
                reg_target = target[:, 4:]
                reg_target[invalid_offsets_mask] = reg_pred[invalid_offsets_mask]

            # 计算损失
            reg_loss += smooth_l1_loss(reg_pred, reg_target)
            cls_loss += focal_loss(cls_pred, cls_target).sum() / num_positives

        # 批次平均
        cls_loss /= valid_imgs
        reg_loss /= valid_imgs

        loss = cls_loss_weight * cls_loss + reg_loss
        return loss, {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'batch_positives': total_positives}

    def nms(self, batch_proposals, batch_attention_matrix, nms_thres, nms_topk, conf_threshold):
        softmax = nn.Softmax(dim=1)
        proposals_list = []
        for proposals, attention_matrix in zip(batch_proposals, batch_attention_matrix):
            anchor_inds = torch.arange(batch_proposals.shape[1], device=proposals.device)
            # NMS过程不需要计算梯度
            with torch.no_grad():
                scores = softmax(proposals[:, :2])[:, 1]
                if conf_threshold is not None:
                    # 应用置信度阈值
                    above_threshold = scores > conf_threshold
                    proposals = proposals[above_threshold]
                    scores = scores[above_threshold]
                    anchor_inds = anchor_inds[above_threshold]
                if proposals.shape[0] == 0:
                    proposals_list.append((proposals[[]], self.anchors[[]], attention_matrix[[]], None))
                    continue
                keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
                keep = keep[:num_to_keep]
            proposals = proposals[keep]
            anchor_inds = anchor_inds[keep]
            attention_matrix = attention_matrix[anchor_inds]
            proposals_list.append((proposals, self.anchors[keep], attention_matrix, anchor_inds))

        return proposals_list

    def proposals_to_pred(self, proposals):
        self.anchor_ys = self.anchor_ys.to(proposals.device)
        self.anchor_ys = self.anchor_ys.double()
        lanes = []
        for lane in proposals:
            lane_xs = lane[5:] / self.img_w
            start = int(round(lane[2].item() * self.n_strips))
            length = int(round(lane[4].item()))
            end = start + length - 1
            end = min(end, len(self.anchor_ys) - 1)
            # 如果提议不是从图像底部开始，将其延伸到x超出图像
            mask = ~((((lane_xs[:start] >= 0.) &
                       (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(np.bool_))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.anchor_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)
            if len(lane_xs) <= 1:
                continue
            points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def decode(self, proposals_list, as_lanes=False):
        softmax = nn.Softmax(dim=1)
        decoded = []
        for proposals, _, _, _ in proposals_list:
            proposals[:, :2] = softmax(proposals[:, :2])
            proposals[:, 4] = torch.round(proposals[:, 4])
            if proposals.shape[0] == 0:
                decoded.append([])
                continue
            if as_lanes:
                pred = self.proposals_to_pred(proposals)
            else:
                pred = proposals
            decoded.append(pred)

        return decoded

    def forward(self, x, conf_threshold=None, nms_thres=0, nms_topk=3000):
        # 1. 特征提取
        x = self.feature_extractor(x)     # (B, 512, 11, 20)
        
        # 2. 特征处理
        x = self.conv1(x)                 # (B, 64, 11, 20)
        x = F.relu(x)
        x = self.conv2(x)                 # (B, 64, 11, 20)
        x = F.relu(x)
        x = self.conv3(x)                 # (B, N, 11, 20)
        
        # 3. 特征重塑
        B = x.shape[0]
        x = x.permute(0, 2, 3, 1)         # (B, 11, 20, N)
        x = x.reshape(B, -1, self.N)      # (B, 220, N)
        x = x.transpose(1, 2)             # (B, N, 220)
        
        # 4. 注意力机制
        attention_features = x.reshape(-1, x.size(-1))  # (B*N, 220)
        scores = self.attention_layer(attention_features)  # (B*N, N-1)
        attention = F.softmax(scores, dim=1).reshape(B, self.N, -1)  # (B, N, N-1)
        
        # 构建注意力矩阵
        attention_matrix = torch.eye(self.N, device=x.device).repeat(B, 1, 1)  # (B, N, N)
        non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        attention_matrix[:] = 0
        attention_matrix[non_diag_inds[:, 0], non_diag_inds[:, 1], non_diag_inds[:, 2]] = attention.flatten()
        
        # 应用注意力
        attention_features = torch.bmm(
            torch.transpose(attention_features.reshape(B, self.N, -1), 1, 2),
            torch.transpose(attention_matrix, 1, 2)
        ).transpose(1, 2)  # (B, N, 220)
        
        # 5. 分类和回归
        cls_logits = self.cls_layer(attention_features.transpose(1, 2)).transpose(1, 2)  # (B, N, 2)
        reg = self.reg_layer(attention_features.transpose(1, 2)).transpose(1, 2)         # (B, N, 73)
        
        # 6. 生成最终输出
        cls_scores = F.softmax(cls_logits, dim=2)
        offsets = self.anchor_template + reg
        
        proposals = torch.cat([cls_scores, self.anchor_xy.expand(B, -1, -1), offsets], dim=2)
        
        # 7. 应用NMS
        if conf_threshold is not None:
            proposals_list = self.nms(proposals, attention_matrix, nms_thres, nms_topk, conf_threshold)
            return proposals_list
        
        return proposals 