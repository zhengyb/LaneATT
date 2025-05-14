import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
import time
from nms import nms

# Predefined 20 distinct colors in BGR format
PREDEFINED_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (192, 192, 192),# Silver
    (64, 64, 64),   # Dark Gray
    (0, 165, 255),  # Orange
    (147, 20, 255), # Pink
    (35, 142, 107), # Forest Green
    (222, 196, 176),# Tan
    (179, 89, 0),   # Brown
    (133, 21, 199)  # Violet
]

def do_nms(proposals, conf_threshold=0.4, nms_thres=50., nms_topk=4):
    """Perform NMS on the proposals."""
    proposals = torch.from_numpy(proposals).cuda()
    scores = proposals[:, 1]
    # apply confidence threshold
    above_threshold = scores > conf_threshold
    proposals = proposals[above_threshold]
    scores = scores[above_threshold]
    
    # cuda implementation
    keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
    keep = keep[:num_to_keep]
    proposals = proposals[keep]
    return proposals

def post_process(img, proposals, n_offsets=72, image_file_path=None):
    """Post process the network output."""
    start_time = time.perf_counter()
    
    # proposals_to_pred
    n_strips = n_offsets - 1
    anchor_ys = torch.linspace(1, 0, steps=n_offsets, dtype=torch.float32, device='cuda:0')
    anchor_ys = anchor_ys.double()
    lanes = []
    
    for lane in proposals:
        lane_xs = lane[5:] / 640
        start = int(round(lane[2].item() * n_strips))
        length = int(round(lane[4].item()))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        
        # if the proposal does not start at the bottom of the image,
        # extend its proposal until the x is outside the image
        mask = ~((((lane_xs[:start] >= 0.) &
                   (lane_xs[:start] <= 1.)).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        lane_ys = anchor_ys[lane_xs >= 0]
        lane_xs = lane_xs[lane_xs >= 0]
        lane_xs = lane_xs.flip(0).double()
        lane_ys = lane_ys.flip(0)
        
        if len(lane_xs) <= 1:
            continue
            
        points = torch.stack((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), dim=1).squeeze(2)
        lanes.append(points.cpu().numpy())

    # Visualize
    img_h, img_w = img.shape[:2]
    for idx, lane_points in enumerate(lanes):
        # Get color using modulo to cycle through predefined colors
        color = PREDEFINED_COLORS[idx % len(PREDEFINED_COLORS)]
        
        # scale back to input size
        lane_points[:, 0] *= img_w
        lane_points[:, 1] *= img_h
        lane_points = lane_points.round().astype(int)
        for point in lane_points:
            cv2.circle(img, tuple(point), 2, color, -1)

    # Save the image with the result
    if image_file_path is not None:
        output_file_path = image_file_path.split('.jpg')[0] + '_result.jpg'
        cv2.imwrite(output_file_path, img)
        print(f"Result saved to {output_file_path}")

    elapsed = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    return img

def inference_on_image(onnx_file_path, image_file_path, benchmark=False):
    """Run inference using ONNX runtime on a single image."""
    # Check if ONNX file exists
    if not os.path.exists(onnx_file_path):
        print(f'ONNX file {onnx_file_path} not found!')
        return
        
    # Create ONNX runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_file_path, providers=providers)
    
    # Load and preprocess image
    image_raw = cv2.imread(image_file_path)
    if image_raw is None:
        print(f'Could not read image {image_file_path}!')
        return
        
    image = cv2.resize(image_raw, (640, 360), cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image.transpose([2, 0, 1])  # HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    if benchmark:
        # Warmup
        for _ in range(10):
            output = session.run(None, {input_name: image})[0]
            
        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            output = session.run(None, {input_name: image})[0]
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            
        avg_time = sum(times) / len(times)
        print(f"┌{'─'*40}┐")
        print(f"│ 推理速度分析 (平均100次)    │")
        print(f"├{'─'*40}┤")
        print(f"│ 推理耗时: {avg_time:>8.2f} ms/帧   │")
        print(f"│ 理论FPS : {1000/avg_time:>8.1f} FPS      │")
        print(f"└{'─'*40}┘")
    else:
        output = session.run(None, {input_name: image})[0]
    
    # Post-processing
    proposals = do_nms(output, conf_threshold=0.5, nms_thres=50., nms_topk=5)
    result_img = post_process(image_raw, proposals, image_file_path=image_file_path)
    
    return result_img

if __name__ == '__main__':
    onnx_file = './LaneATT_r18_tusimple-0513.onnx'
    image_file = './datasets/tusimple_test_image/3.jpg'
    
    print("ONNX Runtime version:", ort.__version__)
    inference_on_image(onnx_file, image_file, benchmark=False) 