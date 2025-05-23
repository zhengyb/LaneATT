# lanedet_on_onnx_cpu.py
# python3 -m venv onnx_infer
# source onnx_infer/bin/activate
# pip install opencv-python numpy onnxruntime scipy
# python3 lanedet_on_onnx_cpu.py

import os
import cv2
import numpy as np
import onnxruntime as ort
import time
import json

from scipy.optimize import linear_sum_assignment


# Global variables and constants
DEVICE_CUDA = 'cuda:0'
DEVICE_CPU = 'cpu'
DEVICE = DEVICE_CPU  # Default to CPU


TUSIMPLE_IMG_RES = (720, 1280)
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

def compute_iou(a, b, threshold):
    """CPU version of IoU computation for lane proposals.
    
    Args:
        a: First lane proposal
        b: Second lane proposal
        threshold: IoU threshold
    Returns:
        bool: True if IoU is below threshold
    """
    N_STRIPS = 71  # N_OFFSETS - 1
    DATASET_OFFSET = 0
    
    # Calculate start and end points
    start_a = int(a[2] * N_STRIPS - DATASET_OFFSET + 0.5)  # 0.5 for rounding
    start_b = int(b[2] * N_STRIPS - DATASET_OFFSET + 0.5)
    start = max(start_a, start_b)
    
    # Calculate end points with adjustment for negative numbers
    end_a = start_a + int(a[4] + 0.5) - 1 - (int(a[4] - 1) < 0)
    end_b = start_b + int(b[4] + 0.5) - 1 - (int(b[4] - 1) < 0)
    end = min(min(end_a, end_b), 71)  # N_OFFSETS - 1
    
    if end < start:
        return False
        
    # Calculate distance
    dist = 0
    for i in range(5 + start, 5 + end + 1):
        dist += abs(float(a[i]) - float(b[i]))
    
    # Return True if below threshold
    return dist < (threshold * (end - start + 1))


def do_nms_cpu(proposals, conf_threshold=0.4, nms_thres=50., nms_topk=4):
    """Perform NMS on the proposals using pure numpy implementation."""
    # Convert input to numpy if it's not already
    if isinstance(proposals, np.ndarray):
        proposals_np = proposals
    else:
        proposals_np = proposals.numpy() if hasattr(proposals, 'numpy') else np.array(proposals)
    
    scores = proposals_np[:, 1]
    #print(f"scores.shape: {scores.shape}")
    # apply confidence threshold
    above_threshold = scores > conf_threshold
    proposals_np = proposals_np[above_threshold]
    scores = scores[above_threshold]
    
    # If no proposals above threshold, return empty array with correct shape
    if len(scores) == 0:
        print(f"No proposals above confidence threshold")
        return proposals_np
    
    # Sort by confidence score
    score_order = np.argsort(-scores)  # descending order
    proposals_np = proposals_np[score_order]
    
    # Initialize arrays for NMS
    keep = []
    N = len(proposals_np)
    parent_object_index = np.zeros(N, dtype=np.int64)
    
    # Process boxes in order of confidence
    for i in range(N):
        # If we've collected enough boxes, break
        if len(keep) >= nms_topk:
            break
            
        # Skip if already marked
        if parent_object_index[i] != 0:
            continue
            
        keep.append(i)
        current_idx = len(keep)  # 1-based index
        
        # Compare with all remaining boxes
        for j in range(i + 1, N):
            if parent_object_index[j] != 0:
                continue
                
            # Check IoU
            if compute_iou(proposals_np[i], proposals_np[j], nms_thres):
                parent_object_index[j] = current_idx
                
        parent_object_index[i] = current_idx
    
    # Keep only the selected proposals
    keep = np.array(keep)
    return proposals_np[keep[:min(len(keep), nms_topk)]]

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
    """Post process the network output using pure numpy implementation."""
    start_time = time.perf_counter()
    
    # Handle empty proposals
    if len(proposals) == 0:
        return []
    
    # Convert proposals to numpy if needed
    if not isinstance(proposals, np.ndarray):
        proposals = proposals.numpy() if hasattr(proposals, 'numpy') else np.array(proposals)
    
    # Create anchor points
    anchor_ys = np.linspace(1, 0, n_offsets, dtype=np.float64)
    n_strips = n_offsets - 1
    lanes = []
    
    # Handle case when no lanes are detected (all proposals have low confidence)
    if proposals.shape[0] == 0:
        return []
    
    for lane in proposals:
        lane_xs = lane[5:] / 640  # Normalize x coordinates
        start = int(round(float(lane[2]) * n_strips))
        length = int(round(float(lane[4])))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        
        # if the proposal does not start at the bottom of the image,
        # extend its proposal until the x is outside the image        
        mask = np.logical_not(
            ((lane_xs[:start] >= 0.0) & (lane_xs[:start] <= 1.0))[::-1].cumprod()[::-1]
        )
        lane_xs[end + 1:] = -2
        lane_xs[:start][mask] = -2
        
        # Keep only the valid points
        valid_indices = lane_xs >= 0
        lane_xs = lane_xs[valid_indices]
        lane_ys = anchor_ys[valid_indices]
        
        # Reverse points to maintain correct order
        lane_xs = lane_xs[::-1]
        lane_ys = lane_ys[::-1]
        
        if len(lane_xs) <= 1:
            continue
            
        # Stack x and y coordinates
        points = np.stack((lane_xs, lane_ys), axis=1)
        lanes.append(points)
    
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
    if DEVICE == DEVICE_CPU:
        providers = ['CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(onnx_file_path, providers=providers)
    #print("Using providers:", session.get_providers())
    
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
        outputs = session.run(None, {input_name: image})
    
    # Post-processing with debug info
    try:
        # convert shape: (1, 1000, 77) to (1000, 77)
        cls_scores = outputs[0]
        reg_proposals = outputs[1]
        #print(f"cls_scores.shape: {cls_scores.shape}")
        #print(f"reg_proposals.shape: {reg_proposals.shape}")
        # concat cls_scores, reg_proposals
        output = np.concatenate([cls_scores, reg_proposals], axis=2)
        output = output.squeeze(0)
        #print(f"output.shape: {output.shape}")
        proposals = do_nms_cpu(output, conf_threshold=0.5, nms_thres=50., nms_topk=4)
        if len(proposals) == 0:
            #print(f"No proposals above confidence threshold for {image_file_path}")
            return [], None if visualize else None
            
        #print(f"proposals.shape: {proposals.shape}")
        lanes = post_process(proposals)
        if len(lanes) == 0:
            print(f"No valid lanes after post-processing for {image_file_path}")
            return [], None if visualize else None
            
    except Exception as e:
        print(f"Error during post-processing: {str(e)}")
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

def draw_lane(lane, img=None, img_shape=None, width=30):
    """Draw a lane (a list of points) on an image by drawing a line with width `width` through each
    pair of points i and i+i"""
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    if len(lane) < 2:
        return img
    lane = lane.astype(np.int32)
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color=(1,), thickness=width)
    return img


def discrete_cross_iou(xs, ys, width=30, img_shape=TUSIMPLE_IMG_RES):
    """For each lane in xs, compute its Intersection Over Union (IoU) with each lane in ys by drawing the lanes on
    an image
    xs: pred lane point list in [(x, y), ...]
    ys: gt lane point list in [(x, y), ...]
    """
    xs = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in xs]
    ys = [draw_lane(lane, img_shape=img_shape, width=width) > 0 for lane in ys]

    ious = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            # IoU by the definition: sum all intersections (binary and) and divide by the sum of the union (binary or)
            ious[i, j] = (x & y).sum() / (x | y).sum()
    return ious

def _culane_metric(pred, anno, raw_file, width=30, iou_threshold=0.4, unofficial=False, img_shape=TUSIMPLE_IMG_RES,
                    draw_img=False, img_path=None):
    """Computes CULane's metric for a single image"""

    pred_ious = np.zeros(len(pred))
    if len(pred) == 0:
        return 0, 0, len(anno), pred_ious, pred_ious > iou_threshold
    if len(anno) == 0:
        return 0, len(pred), 0, pred_ious, pred_ious > iou_threshold
      
    pred = np.array([np.array(pred_lane) for pred_lane in pred], dtype=object)
    anno = np.array([np.array(anno_lane) for anno_lane in anno], dtype=object)

    ious = discrete_cross_iou(pred, anno, width=width, img_shape=img_shape)
    #print("ious:")
    #print(ious)
    row_ind, col_ind = linear_sum_assignment(1 - ious)

    #debug
    #print("mached ious:")
    #print(ious[row_ind, col_ind])
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp # 误判的车道线
    fn = len(anno) - tp # 漏判的车道线
    pred_ious[row_ind] = ious[row_ind, col_ind]
    #print("pred_ious:")
    #print(pred_ious)
    #print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    return tp, fp, fn, pred_ious, pred_ious > iou_threshold    

class LaneEval:
    @staticmethod
    def bench_one_submit_f1(pred_file, gt_file, save_img=True):
        """
        bench_one_submit_f1 函数用于计算预测车道线与真实车道线的F1得分。
        input:
            pred_file: 预测的车道线
            gt_file: 真实的车道线
        output:
            f1: f1得分
        """
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        try:
            json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        except BaseException as e:
            print(f"Fail to load json file of the ground truth: {e}")
            raise Exception('Fail to load json file of the ground truth.')
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {img['raw_file']: img for img in json_gt}
        total_tp, total_fp, total_fn = 0., 0., 0.
        run_times = []
        predictions = []
        annotations = []
        raw_files = []
        for pred in json_pred:
            # for each image
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']

            # format gt_lanes & pred_lanes
            iou_gt_lanes = []
            for lane in gt_lanes:
                lane_ious = [(x, y) for x, y in zip(lane, y_samples) if x >= 0]
                if len(lane_ious) < 2:
                    continue
                iou_gt_lanes.append(lane_ious)
            iou_pred_lanes = []
            for lane in pred_lanes:
                lane_ious = [(x, y) for x, y in zip(lane, y_samples) if x >= 0]
                if len(lane_ious) < 2:
                    continue
                iou_pred_lanes.append(lane_ious)
            predictions.append(iou_pred_lanes)
            annotations.append(iou_gt_lanes)
            if save_img:
                raw_files.append(raw_file)
            else:
                raw_files.append(None)
        results = []
        idx = 0
        for pred, anno, raw_file in zip(predictions, annotations, raw_files):
            if idx % 2 == 0:
                print("\r\\ {}".format(idx), end="", flush=True)
            else:
                print("\r/ {}".format(idx), end="", flush=True)
            results.append(_culane_metric(pred, anno, raw_file, width=30, unofficial=False, img_shape=TUSIMPLE_IMG_RES))
            idx += 1
        print("")
        num = len(gts)
        total_tp = sum(tp for tp, _, _, _, _ in results)
        total_fp = sum(fp for _, fp, _, _, _ in results)
        total_fn = sum(fn for _, _, fn, _, _ in results)
        if total_tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = float(total_tp) / (total_tp + total_fp)
            recall = float(total_tp) / (total_tp + total_fn)
            f1 = 2 * precision * recall / (precision + recall)      
        return json.dumps([{
            'name': 'F1',
            'value': f1,
            'order': 'desc'
        }, {
            'name': 'Precision',
            'value': precision,
            'order': 'desc'
        }, {
            'name': 'Recall',
            'value': recall,
            'order': 'desc'
        }, {
            'name': 'FPS',
            'value': 1000. / np.mean(run_times),
            'order': 'desc'
        }, {
            'name': 'TP',
            'value': total_tp,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': total_fp,
            'order': 'desc'
        }, {
            'name': 'FN',
            'value': total_fn,
            'order': 'desc'
        }])



# Expected metrics:      {'F1': 0.8252276260270932, 'Precision': 0.869443144595227, 'Recall': 0.7852916314454776, 'FPS': 1000.0, 'TP': 1858, 'FP': 279, 'FN': 508}
# Result metrics(GPU):   {'F1': 0.8249113475177304, 'Precision': 0.8671947809878844, 'Recall': 0.7865595942519019, 'FPS': 1000.0, 'TP': 1861, 'FP': 285, 'FN': 505}
# Result metrics(CPU):   {'F1': 0.8208425720620842, 'Precision': 0.863339552238806, 'Recall': 0.7823330515638208, 'FPS': 1000.0, 'TP': 1851, 'FP': 293, 'FN': 515}
# Result (CPU+local LaneEval):          {'F1': 0.825354609929078, 'Precision': 0.8676607642124884, 'Recall': 0.7869822485207101, 'FPS': 1000.0, 'TP': 1862, 'FP': 284, 'FN': 504}
# Result (CPU+local LaneEval):          {'F1': 0.8249113475177304, 'Precision': 0.8671947809878844, 'Recall': 0.7865595942519019, 'FPS': 1000.0, 'TP': 1861, 'FP': 285, 'FN': 505}
# Result 0521: {'F1': 0.8249113475177304, 'Precision': 0.8671947809878844, 'Recall': 0.7865595942519019, 'FPS': 1000.0, 'TP': 1861, 'FP': 285, 'FN': 505}
# RKNN FP16 simulator: {'F1': 0.8238575413001501, 'Precision': 0.8366013071895425, 'Recall': 0.8114961961115807, 'FPS': 1000.0, 'TP': 1920, 'FP': 375, 'FN': 446}
# RKNN simulator INT8 hybrid, ecu_thr=200, optlvl=0: {'F1': 0.7910064239828695, 'Precision': 0.8016493055555556, 'Recall': 0.7806424344885884, 'FPS': 1000.0, 'TP': 1847, 'FP': 457, 'FN': 519}
# RKNN simulator INT8 hybrid, ecu_thr=200, optlvl=3: {'F1': 0.7910064239828695, 'Precision': 0.8016493055555556, 'Recall': 0.7806424344885884, 'FPS': 1000.0, 'TP': 1847, 'FP': 457, 'FN': 519}
def validate_onnx_model(onnx_file_path, dataset_anno_path):
    annotations = []
    pred_list = []
    # Load dataset annotations
    with open(dataset_anno_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    
    images_dir = os.path.dirname(dataset_anno_path)
    print(f"Predicting on {DEVICE}...")
    # Process each image in the dataset
    pred_time_start = time.time()
    for anno_idx in range(len(annotations)):
        if anno_idx > 1000:
            break
        if anno_idx % 2 == 0:
            print("\r\\ {}".format(anno_idx), end="", flush=True)
            pass
        else:
            print("\r/ {}".format(anno_idx), end="", flush=True)
            pass
        anno = annotations[anno_idx]
        image_file = os.path.join(images_dir, anno['raw_file'])
        try:
            lanes, result_img = inference_on_image(onnx_file_path, image_file, benchmark=False, visualize=False)
            #print(f"Inference {anno['raw_file']}, detected {len(lanes)} lanes")
            
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
            continue
    pred_time_end = time.time()
    print(f"FPS: {len(annotations) / (pred_time_end - pred_time_start)}")
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
    result = json.loads(LaneEval.bench_one_submit_f1('pred_list.json', dataset_anno_path))
    print("Metrics:")
    metrics = {}
    for ret in result:
        metrics[ret['name']] = ret['value']
    print(metrics)
    print("Validation done")

    
if __name__ == '__main__':
    onnx_file = './LaneATT_r18_tusimple-0513.onnx'
    onnx_file = './LaneATT_r18_tusimple-0519.onnx'
    #onnx_file = './LaneATT_test.sim-b5-attention-2.onnx'
    onnx_file = './LaneATT_test.sim-bim.onnx'
    onnx_file = './LaneATT_test.sim-nobim.onnx'
    #onnx_file = './LaneATT_test.sim-org.onnx'
    onnx_file = './LaneATT_test.sim-bim1d.onnx'
    onnx_file = './LaneATT_test.sim-bvm.onnx'
    onnx_file = './LaneATT_test.sim-bidx1d.onnx'
    onnx_file = './LaneATT_test.sim-bim3d.onnx'
    onnx_file = './LaneATT_test.sim-2outputs.onnx'
    onnx_file = './LaneATT_test.sim-2outputs-2.onnx'
    # Display available providers
    print("Available ONNX Runtime providers:", ort.get_available_providers())
    print(f"Using device: {DEVICE}")

    print(f"onnx_file: {onnx_file}")
    
    if False:
        image_file = 'datasets/sampled_tusimple/images/val/000012.jpg'
        image_file = 'datasets/tusimple_test_image/0.jpg'
        print("ONNX Runtime version:", ort.__version__)
        lanes, result_img = inference_on_image(onnx_file, image_file, benchmark=False, visualize=True) 
        print(f"Inference done, detected {len(lanes)} lanes")

    if True:
        dataset_anno_path = 'datasets/sampled_tusimple/sampled_anno_val.json'
        print("Validate onnx model")
        validate_onnx_model(onnx_file, dataset_anno_path)
        #result = json.loads(LaneEval.bench_one_submit_f1('pred_list.json', dataset_anno_path))
        #print("Metrics:")
        #metrics = {}
        #for ret in result:
        #    metrics[ret['name']] = ret['value']
        #print(metrics)
        print("Validate onnx model done")