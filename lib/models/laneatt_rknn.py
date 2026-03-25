import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

class LaneATT_RKNN_Friendly(nn.Module):
    def __init__(self, cls_num=2, reg_num=73, S=72, img_w=640, img_h=360, anchor_feat_channels=64):
        super().__init__()
        self.cls_num = cls_num
        self.reg_num = reg_num
        self.img_w = img_w
        self.img_h = img_h
        self.n_strips = S - 1
        self.n_offsets = S
        self.stride = 32  # ResNet18的stride
        self.fmap_h = img_h // self.stride
        self.fmap_w = img_w // self.stride
        self.anchor_feat_channels = anchor_feat_channels

        # Anchor angles, same ones used in Line-CNN
        self.left_angles = [72., 60., 49., 39., 30., 22.]
        self.right_angles = [108., 120., 131., 141., 150., 158.]
        self.bottom_angles = [165., 150., 141., 131., 120., 108., 100., 90., 80., 72., 60., 49., 39., 30., 15.]

        # Generate anchors
        self.anchors, self.anchors_cut = self.generate_anchors(lateral_n=72, bottom_n=128)
        self.N = len(self.anchors)  # number of anchors

        # --- Feature extractor: ResNet18 + 1x1 Conv ---
        backbone = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])  # output: (B, 512, 11, 20)
        self.conv1 = nn.Conv2d(512, anchor_feat_channels, kernel_size=1)  # (B, 64, 11, 20)

        # --- Anchor Feature Extractor ---
        self.anchor_conv = nn.Conv2d(
            in_channels=anchor_feat_channels, out_channels=self.N, kernel_size=1, bias=False
        )

        # --- Regression & Classification Heads ---
        self.cls_layer = nn.Conv1d(in_channels=704, out_channels=cls_num, kernel_size=1)
        self.reg_layer = nn.Conv1d(in_channels=704, out_channels=reg_num, kernel_size=1)

        # --- Anchor Offsets ---
        self.register_buffer("anchor_xy", self.anchors[:, 2:4].unsqueeze(0))        # (1, N, 2)
        self.register_buffer("anchor_template", self.anchors[:, 4:].unsqueeze(0))   # (1, N, 73)

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
