import torch
import torch.nn as nn

from lib.models.laneatt import LaneATT


class LaneATTSimplifiedONNX(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fmap_h = model.fmap_h
        self.fmap_w = model.fmap_w
        self.anchor_feat_channels = model.anchor_feat_channels
        self.anchors = model.anchors

        # Register static index/mask as buffer for ONNX compatibility
        self.register_buffer('cut_xs', model.cut_xs)
        self.register_buffer('cut_ys', model.cut_ys)
        self.register_buffer('cut_zs', model.cut_zs)
        self.register_buffer('invalid_mask', model.invalid_mask.float())  # (1000, 1)

        # Layers from base model
        self.feature_extractor = model.feature_extractor
        self.conv1 = model.conv1
        self.cls_layer = model.cls_layer
        self.reg_layer = model.reg_layer

        # Add reduce layer to reduce final feature dimensions
        self.reduce_layer = nn.Linear(self.anchor_feat_channels * 2, 128)

    def forward(self, x):
        batch_features = self.feature_extractor(x)        # (1, C, H, W)
        batch_features = self.conv1(batch_features)       # (1, C1, H, W)
        batch_features = batch_features[0]                # Remove batch dim -> (C1, H, W)

        # Reshape and gather anchor features
        feat = batch_features.view(self.anchor_feat_channels, -1).transpose(0, 1)  # (H*W, C)
        
        # Ensure anchor indices are within bounds
        anchor_indices = self.cut_xs + self.fmap_w * self.cut_ys + self.fmap_w * self.fmap_h * self.cut_zs
        anchor_indices = torch.clamp(anchor_indices, 0, self.fmap_h * self.fmap_w - 1)
        
        # Gather features and apply mask
        anchor_features = feat[anchor_indices]  # (1000, C)
        
        # Reshape invalid_mask to match anchor_features dimensions
        invalid_mask = self.invalid_mask.expand(-1, self.anchor_feat_channels)  # (1000, C)
        anchor_features = anchor_features * (1.0 - invalid_mask)

        # Simplified attention (dot-product based)
        scores = torch.matmul(anchor_features, anchor_features.T) / (self.anchor_feat_channels ** 0.5)  # (1000, 1000)
        attention = torch.softmax(scores, dim=1)
        attended = torch.matmul(attention, anchor_features)  # (1000, C)

        # Concatenate + reduce
        combined = torch.cat((anchor_features, attended), dim=1)  # (1000, C*2)
        reduced = self.reduce_layer(combined)  # (1000, 128)

        # Predict
        cls_logits = self.cls_layer(reduced)  # (1000, 2)
        reg = self.reg_layer(reduced)         # (1000, 73)

        # Softmax + merge with anchor
        softmax = nn.Softmax(dim=1)
        cls_probs = softmax(cls_logits)
        proposals = torch.cat([cls_probs, self.anchors[:, 2:4], self.anchors[:, 4:] + reg], dim=1)

        return proposals


def export_onnx(onnx_file_path):
    backbone_name = 'resnet18'
    checkpoint_file_path = 'experiments/laneatt_r18_tusimple/backup_models/model_0035.pt'
    anchors_freq_path = 'data/tusimple_250418pm_anchors_mask.pt'

    model = LaneATT(backbone=backbone_name, anchors_freq_path=anchors_freq_path, topk_anchors=1000)
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Wrap in simplified module
    onnx_model = LaneATTSimplifiedONNX(model)
    onnx_model.eval()

    dummy_input = torch.randn(1, 3, 360, 640)
    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_file_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch'},   # batch dimension (optional for RKNN)
            'output': {0: 'num_anchors'}
        }
    )
    print(f"ONNX model exported to {onnx_file_path}")

if __name__ == '__main__':
    export_onnx('./LaneATT_r18_tusimple-0519-3.onnx')