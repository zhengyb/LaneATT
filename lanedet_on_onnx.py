import os
import cv2
import torch
import numpy as np
import onnxruntime as ort
import time
from nms import nms
import json

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


ANCHOR_YS = torch.linspace(1, 0, steps=72, dtype=torch.float32, device='cuda:0')
ANCHOR_YS = ANCHOR_YS.double()

def do_nms(proposals, conf_threshold=0.4, nms_thres=50., nms_topk=4):
    """Perform NMS on the proposals."""
    proposals = torch.from_numpy(proposals).cuda()
    scores = proposals[:, 1]
    
    # apply confidence threshold
    above_threshold = scores > conf_threshold
    proposals = proposals[above_threshold]
    scores = scores[above_threshold]
    
    # If no proposals above threshold, return empty tensor with correct shape
    if len(scores) == 0:
        return proposals  # proposals will be empty but with correct shape
    
    # cuda implementation
    keep, num_to_keep, _ = nms(proposals, scores, overlap=nms_thres, top_k=nms_topk)
    keep = keep[:num_to_keep]
    proposals = proposals[keep]
    return proposals

def visualize_lanes(img, lanes, image_file_path=None):
    """Visualize detected lanes on the image."""
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

    return img

def post_process(proposals, n_offsets=72):
    """Post process the network output."""
    start_time = time.perf_counter()
    
    # Handle empty proposals
    if len(proposals) == 0:
        return []
    
    # proposals_to_pred
    n_strips = n_offsets - 1
    anchor_ys = ANCHOR_YS
    lanes = []
    
    # Handle case when no lanes are detected (all proposals have low confidence)
    if proposals.shape[0] == 0:
        return []
    
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
    
    elapsed = (time.perf_counter() - start_time) * 1000  # Convert to milliseconds
    # print(f"Post-processing time: {elapsed:.2f}ms")
    
    return lanes

def inference_on_image(onnx_file_path, image_file_path, benchmark=False, visualize=False):
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
    
    # Post-processing with debug info
    try:
        proposals = do_nms(output, conf_threshold=0.5, nms_thres=50., nms_topk=4)
        if len(proposals) == 0:
            #print(f"No proposals above confidence threshold for {image_file_path}")
            return [], None if visualize else None
            
        lanes = post_process(proposals)
        if len(lanes) == 0:
            print(f"No valid lanes after post-processing for {image_file_path}")
            return [], None if visualize else None
            
    except Exception as e:
        print(f"Error during post-processing: {str(e)}")
        torch.cuda.empty_cache()  # Try to recover GPU memory
        return [], None if visualize else None
    
    if visualize:
        result_img = visualize_lanes(image_raw.copy(), lanes, image_file_path=image_file_path)
    else:
        result_img = None

    return lanes, result_img

def pred2lanes(pred, y_samples, img_h, img_w):
    """Convert lane predictions to TuSimple format.
    
    Args:
        pred: List of lane points, each lane is a numpy array of shape (N, 2) with normalized coordinates
        y_samples: List of y coordinates to sample
        img_h: Original image height
        img_w: Original image width
    Returns:
        lanes: List of lanes in TuSimple format
    """
    # Convert y_samples to normalized coordinates
    y_samples = np.array(y_samples, dtype=np.float32) / img_h
    
    lanes = []
    # Handle empty predictions
    if not pred:
        return lanes
        
    for lane_points in pred:
        # Convert lane_points to numpy array if it isn't already
        lane_points = np.array(lane_points)
        
        # Skip if lane has too few points
        if len(lane_points) < 2:
            continue
            
        # Interpolate x coordinates at y_samples
        lane_xs = []
        for y in y_samples:
            # Find the two points that bracket this y
            above_idx = np.where(lane_points[:, 1] <= y)[0]
            below_idx = np.where(lane_points[:, 1] > y)[0]
            
            if len(above_idx) == 0 or len(below_idx) == 0:
                # y is outside the range of this lane
                lane_xs.append(-2)
                continue
                
            # Get the bracketing points
            p1 = lane_points[above_idx[-1]]
            p2 = lane_points[below_idx[0]]
            
            # Interpolate
            x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
            
            # Convert normalized x to pixel coordinates and handle out of bounds
            x_px = int(x * img_w)
            if x_px < 0 or x_px >= img_w:
                x_px = -2
            
            lane_xs.append(x_px)
            
        lanes.append(lane_xs)
    
    return lanes

# Expected metrics: {'F1': 0.8252276260270932, 'Precision': 0.869443144595227, 'Recall': 0.7852916314454776, 'FPS': 1000.0, 'TP': 1858, 'FP': 279, 'FN': 508}
# Result metrics:   {'F1': 0.8249113475177304, 'Precision': 0.8671947809878844, 'Recall': 0.7865595942519019, 'FPS': 1000.0, 'TP': 1861, 'FP': 285, 'FN': 505}
def validate_onnx_model(onnx_file_path, dataset_anno_path, is_carla=False, rgb=False):
    annotations = []
    pred_list = []
    # Load dataset annotations
    with open(dataset_anno_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    if is_carla:
        images_dir_root = dataset_anno_path.split('tusimple_merged')[0]
        images_dir = os.path.dirname(images_dir_root)
    else:
        images_dir = os.path.dirname(dataset_anno_path)
    print("Predicting...")
    # Process each image in the dataset
    for anno_idx in range(len(annotations)):
        if anno_idx > 1000:
            break
        if anno_idx % 2 == 0:
            print("\r\\ {}".format(anno_idx), end="", flush=True)
        else:
            print("\r/ {}".format(anno_idx), end="", flush=True)
        anno = annotations[anno_idx]
        image_file = os.path.join(images_dir, anno['raw_file'])
        try:
            lanes, result_img = inference_on_image(onnx_file_path, image_file, benchmark=False, visualize=False)
            #print(f"Inference {anno['raw_file']}, detected {len(lanes)} lanes")
            # Clear GPU memory after each image
            torch.cuda.empty_cache()
            
            # Create prediction in TuSimple format
            pred = {}
            pred['raw_file'] = anno['raw_file']
            pred['h_samples'] = anno['h_samples']
            # Directly pass the lanes list to pred2lanes without converting to numpy array
            pred['lanes'] = pred2lanes(lanes, anno['h_samples'], 720, 1280)
            pred['run_time'] = 1.           
            pred_list.append(pred)
            
        except Exception as e:
            print(f"Error processing {anno['raw_file']}: {str(e)}")
            # Try to recover and continue with next image
            torch.cuda.empty_cache()
            continue

    print("Saving predictions...")
    # Save predictions
    with open('pred_list.json', 'w') as f:
        for pred in pred_list:
            try:    
                f.write(json.dumps(pred) + '\n')
            except Exception as e:
                print(f"Error writing {pred['raw_file']}: {str(e)}")

    # Uncomment to validate predictions
    print("Validating predictions...")
    from utils.tusimple_metric import LaneEval
    result = json.loads(LaneEval.bench_one_submit_f1('pred_list.json', dataset_anno_path))
    metrics = {}
    for ret in result:
        metrics[ret['name']] = ret['value']
    print(metrics)
    print("Validation done")
    return metrics

def generate_anno_path_list(anno_dir_root, split):
    """Generate a list of annotation file paths for the given split."""
    anno_files = []
    for root, _, files in os.walk(anno_dir_root):
        for file in files:
            if file.endswith('.json') and split in file:
                anno_files.append(os.path.join(root, file))
    return anno_files

if __name__ == '__main__':
    #onnx_file = './onnx/LaneATT_r18_tusimple-0513.onnx'
    onnx_file = './onnx/LaneATT_test-0529RGB.sim.onnx'

    if False:
        
        image_file0 = './datasets/sampled_tusimple/images/val/000049.jpg' # 0 lanes

        image_file2 = './datasets/sampled_tusimple/images/val/000048.jpg' # 2 lanes
        image_file3 = './datasets/sampled_tusimple/images/val/000050.jpg' # 3 lanes

        image_list = [image_file2, image_file0, image_file3]
        for image_file in image_list:
            print(f"Inference {image_file}")
            try:
                lanes, result_img = inference_on_image(onnx_file, image_file, benchmark=False, visualize=True) 
                print(f"Inference {image_file} done, detected {len(lanes)} lanes")
            except Exception as e:
                print(f"Inference {image_file} failed, error: {e}")


    if False:
        print("ONNX Runtime version:", ort.__version__)
        for i in range(10):
            print(f"Inference {i} times")
        lanes, result_img = inference_on_image(onnx_file, image_file, benchmark=False, visualize=False) 
        print(f"Inference {i} times done, detected {len(lanes)} lanes")


    if False:
        dataset_anno_path = 'datasets/sampled_tusimple/sampled_anno_val.json'
        print("Validate onnx model")
        validate_onnx_model(onnx_file, dataset_anno_path)
        print("Validate onnx model done")

    if False:
        metrics_list = []
        anno_dir_root = 'datasets/tusimple-0325/tusimple_merged/'
        split = 'test'
        anno_files = generate_anno_path_list(anno_dir_root, split)
        #anno_files = [anno_files[0], anno_files[1]]  # Only validate on the first two annotation files for now
        print(f"Generated {len(anno_files)} annotation file paths for split '{split}'")
        for anno_file in anno_files:
            print(f"Validating on {anno_file}...")
            metrics = validate_onnx_model(onnx_file, anno_file, is_carla=True)
            metrics_list.append(metrics)
            print(f"Validation on {anno_file} done")

        total_metrics = {}
        for metrics in metrics_list:
            for key in ('TP', 'FP', 'FN'):
                if key not in total_metrics:
                    total_metrics[key] = 0
                value = metrics.get(key, 0)
                total_metrics[key] += value
        
        # recalculate F1, Precision, Recall, etc. based on total TP, FP, FN
        total_TP = total_metrics.get('TP', 0)
        total_FP = total_metrics.get('FP', 0)
        total_FN = total_metrics.get('FN', 0)
        total_metrics['Precision'] = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        total_metrics['Recall'] = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        total_metrics['F1'] = 2 * total_metrics['Precision'] * total_metrics['Recall'] / (total_metrics['Precision'] + total_metrics['Recall']) if (total_metrics['Precision'] + total_metrics['Recall']) > 0 else 0
        total_metrics['FPS'] = 1000.0  # Placeholder, since we are not measuring FPS here

        print(f"Total Metrics: {json.dumps(total_metrics, indent=4)}")
    