import torch
from lib.models.laneatt import LaneATT

USE_ATTENTION = True


class LaneATTONNX(torch.nn.Module):
    def __init__(self, model):
        super(LaneATTONNX, self).__init__()
        # Params
        self.fmap_h = model.fmap_h  # 11
        self.fmap_w = model.fmap_w  # 20
        print(f"self.fmap_h: {self.fmap_h}, self.fmap_w: {self.fmap_w}")
        self.anchor_feat_channels = model.anchor_feat_channels  # 64
        self.anchors = model.anchors
        self.cut_xs = model.cut_xs
        self.cut_ys = model.cut_ys
        self.cut_zs = model.cut_zs
        self.invalid_mask = model.invalid_mask
        # Layers
        self.feature_extractor = model.feature_extractor
        self.conv1 = model.conv1
        self.cls_layer = model.cls_layer
        self.reg_layer = model.reg_layer
        self.attention_layer = model.attention_layer

        # Exporting the operator eye to ONNX opset version 11 is not supported
        attention_matrix = torch.eye(1000)
        self.non_diag_inds = torch.nonzero(attention_matrix == 0.0, as_tuple=False)
        self.non_diag_inds = (
            self.non_diag_inds[:, 1] + 1000 * self.non_diag_inds[:, 0]
        )  # 999000
        self.non_diag_inds = self.non_diag_inds.contiguous()

        #[704000, 1]
        #self.indices = self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs
        #print(f"self.indices.shape: {self.indices.shape}")
        #indices = self.cut_xs + self.fmap_w * self.cut_ys + self.fmap_h * self.fmap_w * self.cut_zs
        #indices = indices.contiguous().view(-1)  # 将indices展平为一维向量
        #print(f"self.indices.shape: {indices.shape}")
        #indices = indices.contiguous()
        #self.register_buffer('indices', indices)
        # register indices as a buffer
        print(f"self.invalid_mask.shape: {self.invalid_mask.shape}")
        # reshape invalid mask to (1, 704000, 1)
        self.reshaped_invalid_mask = model.invalid_mask.view(1, -1, 1)
        print(f"self.reshaped_invalid_mask.shape: {self.reshaped_invalid_mask.shape}")
        #self.reshaped_valid_mask = torch.logical_not(self.reshaped_invalid_mask)

    def simple_cut_features(self, batch_features):
        indices = self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs
        # 使用预定义的索引选择特定位置的特征
        #batch_anchor_features = batch_features[:, indices].\
        #    view(-1, 1000, self.anchor_feat_channels, self.fmap_h, 1)
        batch_anchor_features = batch_features[:, indices]
        return batch_anchor_features

    def forward(self, x):
        batch_features = self.feature_extractor(x)
        batch_features = self.conv1(batch_features)
        # batch_anchor_features = self.cut_anchor_features(batch_features)
        # batchx15360
        # 将特征重塑为一维向量
        batch_anchor_features = batch_features.reshape(-1, int(batch_features.numel()))
        #b1
        # h, w = batch_features.shape[2:4]  # 12, 20
        # indices = self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs
        # 使用预定义的索引选择特定位置的特征
        # batch_anchor_features = batch_anchor_features[:, self.indices].\
        #     view(-1, 1000, self.anchor_feat_channels, self.fmap_h, 1)

        # 使用simple_cut_features处理特征
        batch_anchor_features = self.simple_cut_features(batch_anchor_features)
        #b2
        
        # bim
        # batch_anchor_features[self.invalid_mask] = 0
        # 应用无效掩码，将无效区域的特征置为0
        #batch_anchor_features = batch_anchor_features * torch.logical_not(
        #    self.invalid_mask
        #)
        # bim1d
        #batch_anchor_features = batch_anchor_features * torch.logical_not(
        #    self.reshaped_invalid_mask
        #)
        # bvm
        #batch_anchor_features = batch_anchor_features * self.reshaped_valid_mask
        # bidx1d
        batch_anchor_features[self.reshaped_invalid_mask] = 0

        # Join proposals from all images into a single proposals features batch
        # batchx1000x704
        # 将特征重组为(batch_size, 1000, anchor_feat_channels * fmap_h)的形状
        batch_anchor_features = batch_anchor_features.view(
            -1, 1000, self.anchor_feat_channels * self.fmap_h
        )
        #b3

        # Add attention features
        softmax = torch.nn.Softmax(dim=2)
        if USE_ATTENTION:
            # batchx1000x999
            scores = self.attention_layer(batch_anchor_features)
            attention = softmax(scores)
            bs, _, _ = scores.shape
            attention_matrix = torch.zeros(bs, 1000 * 1000, device=x.device)
            attention_matrix[:, self.non_diag_inds] = attention.reshape(
                -1, int(attention.numel())
            )
            attention_matrix = attention_matrix.view(-1, 1000, 1000)
            attention_features = torch.matmul(
                torch.transpose(batch_anchor_features, 1, 2),
                torch.transpose(attention_matrix, 1, 2),
            ).transpose(1, 2)
            batch_anchor_features = torch.cat(
                (attention_features, batch_anchor_features), dim=2
            )
        else:
            batch_anchor_features = torch.cat(
                (batch_anchor_features, batch_anchor_features), dim=2
            )
            bs = batch_anchor_features.shape[0]

        #b4
    
        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        xs, ys = map(int, self.anchors.shape)
        anchors = self.anchors[None].expand(bs, xs, ys)

        # Add offsets to anchors (1000, 2+2+73)
        reg_proposals = torch.cat(
            [softmax(cls_logits), anchors[:, :, 2:4], anchors[:, :, 4:] + reg], dim=2
        )

        #b5
        return reg_proposals


def export_onnx(onnx_file_path):
    # e.g. laneatt_r18_culane
    backbone_name = "resnet18"
    checkpoint_file_path = (
        "experiments/laneatt_r18_tusimple/backup_models/model_0035.pt"
    )
    anchors_freq_path = "data/tusimple_250418pm_anchors_mask.pt"

    # Load specified checkpoint
    model = LaneATT(
        backbone=backbone_name, anchors_freq_path=anchors_freq_path, topk_anchors=1000
    )
    checkpoint = torch.load(checkpoint_file_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # Export to ONNX
    onnx_model = LaneATTONNX(model)
    dummy_input = torch.randn(1, 3, 360, 640)
    torch.onnx.export(
        onnx_model,
        dummy_input,
        onnx_file_path,
        input_names=["images"],
        output_names=["output"],
    )

    import onnx

    model_onnx = onnx.load(onnx_file_path)

    # Simplify
    try:
        import onnxsim

        print(f"simplifying with onnxsim {onnxsim.__version__}...")
        model_onnx, check = onnxsim.simplify(model_onnx)
        assert check, "Simplified ONNX model could not be validated"
    except Exception as e:
        print(f"simplifier failure: {e}")

    onnx.save(model_onnx, "LaneATT_test.sim-bidx1d.onnx")
    print(f"simplify done. onnx model save in LaneATT_test.sim-bidx1d.onnx")


if __name__ == "__main__":
    export_onnx("./LaneATT_test.onnx")
