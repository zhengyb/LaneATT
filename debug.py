import torch

def test_attention_matrix(batch_anchor_features, attention_layer, softmax):
    # batch_anchor_features.shape: [1, 1000, 999]
    if True:
        # Exporting the operator eye to ONNX opset version 11 is not supported
        constt_attention_matrix = torch.eye(1000)
        non_diag_inds = torch.nonzero(constt_attention_matrix == 0.0, as_tuple=False)
        non_diag_inds = (
            non_diag_inds[:, 1] + 1000 * non_diag_inds[:, 0]
        )  # 999000
        non_diag_inds = non_diag_inds.contiguous()

        scores = attention_layer(batch_anchor_features)
        attention = softmax(scores)
        print(f"attention.shape: {attention.shape}")
        #bs, _, _ = scores.shape
        bs = 1
        print(f"bs: {bs}")
        attention_matrix = torch.zeros(1, 1000 * 1000, device=x.device)
        # ScatterND
        attention_matrix[:, non_diag_inds] = attention.reshape(
                -1, int(attention.numel())
        )
    attention_matrix = attention_matrix.view(-1, 1000, 1000)

    return attention_matrix


def test_attention_matrix_nd(batch_anchor_features, attention_layer, softmax):
    # Create attention matrix directly in 1x1000x1000 shape
    attention_matrix = torch.zeros(1, 1000, 1000, device=batch_anchor_features.device)
    
    # Get attention scores and apply softmax
    scores = attention_layer(batch_anchor_features)
    attention = softmax(scores)
    
    # Create mask for non-diagonal elements
    mask = ~torch.eye(1000, dtype=torch.bool, device=batch_anchor_features.device)
    
    # Reshape attention to match the non-diagonal elements
    attention = attention.reshape(1, 1000, 1000)
    
    # Apply attention values to non-diagonal positions
    attention_matrix[mask] = attention[mask]
    
    return attention_matrix