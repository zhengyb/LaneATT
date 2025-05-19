import torch
import torch.nn as nn


class LaneATTONNX(nn.Module):
    def __init__(self, model, top_k=10):
        super(LaneATTONNX, self).__init__()
        self.fmap_h = model.fmap_h
        self.fmap_w = model.fmap_w
        self.anchor_feat_channels = model.anchor_feat_channels
        self.anchors = model.anchors
        self.cut_xs = model.cut_xs
        self.cut_ys = model.cut_ys
        self.cut_zs = model.cut_zs
        self.invalid_mask = model.invalid_mask
        self.top_k = top_k

        self.feature_extractor = model.feature_extractor
        self.conv1 = model.conv1
        self.cls_layer = model.cls_layer
        self.reg_layer = model.reg_layer
        self.attention_layer = model.attention_layer  # 仍保留以兼容权重

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.conv1(features)

        # flatten and extract anchor features
        flat_feat = features[0].flatten()
        anchor_feat = flat_feat[self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs]
        anchor_feat = anchor_feat.view(1000, self.anchor_feat_channels, self.fmap_h, 1)

        # mask invalid anchors
        anchor_feat = anchor_feat * torch.logical_not(self.invalid_mask)
        anchor_feat = anchor_feat.view(-1, self.anchor_feat_channels * self.fmap_h)  # [1000, C]

        # Top-K attention
        n = anchor_feat.size(0)
        attention_scores = torch.matmul(anchor_feat, anchor_feat.T)
        diag_mask = torch.eq(torch.arange(n).view(-1, 1), torch.arange(n).view(1, -1)).to(attention_scores.device)
        attention_scores = attention_scores - 1e9 * diag_mask.float()


        topk_scores, topk_indices = torch.topk(attention_scores, self.top_k, dim=-1)  # [1000, K]
        topk_weights = torch.softmax(topk_scores, dim=-1)  # [1000, K]

        topk_features = anchor_feat[topk_indices]  # [1000, K, C]
        attention_features = torch.sum(topk_features * topk_weights.unsqueeze(-1), dim=1)  # [1000, C]

        # concat attention-enhanced features
        anchor_feat = torch.cat([attention_features, anchor_feat], dim=1)  # [1000, 2C]

        # classification and regression
        cls_logits = self.cls_layer(anchor_feat)
        reg = self.reg_layer(anchor_feat)

        # proposal = softmax + anchors + regression offsets
        softmax = nn.Softmax(dim=1)
        reg_proposals = torch.cat([
            softmax(cls_logits),           # class prob
            self.anchors[:, 2:4],          # anchor offsets
            self.anchors[:, 4:] + reg      # lane offsets
        ], dim=1)

        return reg_proposals  # [1000, 77]

def export_onnx(onnx_file_path):
    from lib.models.laneatt import LaneATT

    # Load model
    backbone_name = 'resnet18'
    checkpoint_file_path = 'experiments/laneatt_r18_tusimple/backup_models/model_0035.pt'
    anchors_freq_path = 'data/tusimple_250418pm_anchors_mask.pt'

    model = LaneATT(backbone=backbone_name, anchors_freq_path=anchors_freq_path, topk_anchors=1000)
    checkpoint = torch.load(checkpoint_file_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Convert wrapper model
    onnx_model = LaneATTONNX(model, top_k=10)
    dummy_input = torch.randn(1, 3, 360, 640)

    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "num_proposals"}},
        verbose=False
    )
    print(f"[✓] ONNX model saved to {onnx_file_path}")



if __name__ == '__main__':
    export_onnx('./LaneATT_r18_tusimple-0519-2.onnx')
