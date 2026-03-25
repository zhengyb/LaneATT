import torch
import torch.nn as nn
import torch.nn.functional as F

class LaneATTONNX_no_gather(nn.Module):
    def __init__(self, cls_num=2, reg_num=73):
        super(LaneATTONNX_no_gather, self).__init__()
        self.cls_num = cls_num
        self.reg_num = reg_num

        # 704 = anchor_feat_channels × fmap_h
        self.cls_layer = nn.Conv1d(in_channels=704, out_channels=cls_num, kernel_size=1)
        self.reg_layer = nn.Conv1d(in_channels=704, out_channels=reg_num, kernel_size=1)

    def forward(self, anchor_features, anchor_xy, anchor_template):
        # anchor_features: [B, 1000, 704]
        # anchor_xy:       [B, 1000, 2]
        # anchor_template: [B, 1000, 73]  # Base anchor sampling points

        x = anchor_features.transpose(1, 2)  # [B, 704, 1000]
        cls_logits = self.cls_layer(x).transpose(1, 2)  # [B, 1000, 2]
        reg = self.reg_layer(x).transpose(1, 2)         # [B, 1000, 73]

        cls_scores = F.softmax(cls_logits, dim=2)       # [B, 1000, 2]
        offsets = anchor_template + reg                 # [B, 1000, 73]

        proposals = torch.cat([cls_scores, anchor_xy, offsets], dim=2)  # [B, 1000, 2+2+73]
        return proposals
