import torch
from lib.models.laneatt import LaneATT

class LaneATTONNX(torch.nn.Module):
    def __init__(self, model):
        super(LaneATTONNX, self).__init__()
        # Params
        self.fmap_h = model.fmap_h  # 11
        self.fmap_w = model.fmap_w  # 20
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
        self.non_diag_inds = torch.nonzero(attention_matrix == 0., as_tuple=False)
        self.non_diag_inds = self.non_diag_inds[:, 1] + 1000 * self.non_diag_inds[:, 0]  # 999000

        #self.indices = self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs   
        self.indices = self.indices.contiguous()

        

    def forward(self, x):
        batch_features = self.feature_extractor(x)
        batch_features = self.conv1(batch_features)
        # batch_anchor_features = self.cut_anchor_features(batch_features)
        # batchx15360
        batch_anchor_features = batch_features.reshape(-1, int(batch_features.numel()))
        # h, w = batch_features.shape[2:4]  # 12, 20     
        indices = self.cut_xs + 20 * self.cut_ys + 12 * 20 * self.cut_zs   
        batch_anchor_features = batch_anchor_features[:, indices].\
            view(-1, 1000, self.anchor_feat_channels, self.fmap_h, 1)        
        # batch_anchor_features[self.invalid_mask] = 0
        batch_anchor_features = batch_anchor_features * torch.logical_not(self.invalid_mask)

        # Join proposals from all images into a single proposals features batch
        # batchx1000x704
        batch_anchor_features = batch_anchor_features.view(-1, 1000, self.anchor_feat_channels * self.fmap_h)

        # Add attention features
        softmax = torch.nn.Softmax(dim=2)
        # batchx1000x999
        scores = self.attention_layer(batch_anchor_features)
        attention = softmax(scores)
        bs, _, _ = scores.shape
        attention_matrix = torch.zeros(bs, 1000 * 1000, device=x.device)
        attention_matrix[:, self.non_diag_inds] = attention.reshape(-1, int(attention.numel()))
        attention_matrix = attention_matrix.view(-1, 1000, 1000)
        attention_features = torch.matmul(torch.transpose(batch_anchor_features, 1, 2),
                                          torch.transpose(attention_matrix, 1, 2)).transpose(1, 2)
        batch_anchor_features = torch.cat((attention_features, batch_anchor_features), dim=2)

        # Predict
        cls_logits = self.cls_layer(batch_anchor_features)
        reg = self.reg_layer(batch_anchor_features)

        xs, ys = map(int, self.anchors.shape)
        anchors = self.anchors[None].expand(bs, xs, ys)

        # Add offsets to anchors (1000, 2+2+73)
        reg_proposals = torch.cat([softmax(cls_logits), anchors[:, :, 2:4], anchors[:, :, 4:] + reg], dim=2)

        return reg_proposals

def export_onnx(onnx_file_path):
    # e.g. laneatt_r18_culane
    backbone_name = 'resnet18'
    checkpoint_file_path = 'experiments/laneatt_r18_culane/models/model_0015.pt'
    anchors_freq_path = 'data/tusimple_250418pm_anchors_mask.pt'

    # Load specified checkpoint
    model = LaneATT(backbone=backbone_name, anchors_freq_path=anchors_freq_path, topk_anchors=1000)
    checkpoint = torch.load(checkpoint_file_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Export to ONNX
    onnx_model = LaneATTONNX(model)
    dummy_input = torch.randn(1, 3, 360, 640)
    torch.onnx.export(
        onnx_model, 
        dummy_input, 
        onnx_file_path, 
        input_names=["images"], 
        output_names=["output"]
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

    onnx.save(model_onnx, "LaneATT_test.sim.onnx")
    print(f"simplify done. onnx model save in LaneATT_test.sim.onnx")   

if __name__ == '__main__':
    export_onnx('./LaneATT_test.onnx')
