import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LaneATT_RKNN_Friendly(nn.Module):
    def __init__(self, anchor_mask_tensor, anchors_tensor, cls_num=2, reg_num=73):
        super().__init__()
        self.cls_num = cls_num
        self.reg_num = reg_num
        self.N = anchor_mask_tensor.shape[0]  # number of anchors, e.g., 1000

        # --- Feature extractor: ResNet18 + 1x1 Conv ---
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # output: (B, 512, 11, 20)
        self.conv1 = nn.Conv2d(512, 64, kernel_size=1)  # (B, 64, 11, 20)

        # --- Anchor Feature Extractor ---
        self.anchor_conv = nn.Conv2d(
            in_channels=64, out_channels=self.N, kernel_size=1, bias=False
        )
        # We will initialize anchor_conv weights using anchor_mask_tensor
        # anchor_mask_tensor shape: (N, 64, 1, 1)
        self.anchor_conv.weight = nn.Parameter(anchor_mask_tensor, requires_grad=False)

        # --- Regression & Classification Heads ---
        self.cls_layer = nn.Conv1d(in_channels=704, out_channels=cls_num, kernel_size=1)
        self.reg_layer = nn.Conv1d(in_channels=704, out_channels=reg_num, kernel_size=1)

        # --- Anchor Offsets ---
        # anchors_tensor shape: (N, 2+2+73)
        self.register_buffer("anchor_xy", anchors_tensor[:, 2:4].unsqueeze(0))        # (1, N, 2)
        self.register_buffer("anchor_template", anchors_tensor[:, 4:].unsqueeze(0))   # (1, N, 73)

    def forward(self, x):  # x: (B, 3, 360, 640)
        x = self.feature_extractor(x)     # → (B, 512, 11, 20)
        x = self.conv1(x)                 # → (B, 64, 11, 20)
        anchor_feat = self.anchor_conv(x) # → (B, N, 11, 20)

        B, N, H, W = anchor_feat.shape
        anchor_feat = anchor_feat.view(B, N, -1)  # → (B, N, 704)
        anchor_feat = anchor_feat.transpose(1, 2) # → (B, 704, N)

        cls_logits = self.cls_layer(anchor_feat).transpose(1, 2)  # (B, N, 2)
        reg = self.reg_layer(anchor_feat).transpose(1, 2)         # (B, N, 73)

        cls_scores = F.softmax(cls_logits, dim=2)
        offsets = self.anchor_template + reg

        proposals = torch.cat([cls_scores, self.anchor_xy.expand(B, -1, -1), offsets], dim=2)  # (B, N, 2+2+73)
        return proposals
