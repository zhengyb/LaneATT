import os
import json
import argparse
import cv2
import torch
import numpy as np
from lib.models.laneatt import LaneATT

# Predefined distinct colors in BGR format
LANE_COLORS = [
    (0, 0, 255),     # Red
    (0, 255, 0),     # Green
    (255, 0, 0),     # Blue
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Magenta
    (255, 255, 0),   # Cyan
    (128, 0, 0),     # Maroon
    (0, 128, 0),     # Dark Green
    (0, 0, 128),     # Navy
    (128, 128, 0),   # Olive
]


def preprocess(img, dst_width=640, dst_height=360):
    img_pre = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB, consistent with training
    img_pre = cv2.resize(img_pre, (dst_width, dst_height))
    img_pre = (img_pre / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre


def draw_lanes(img, pred):
    """Draw detected lanes on image with distinct colors."""
    for idx, lane in enumerate(pred):
        color = LANE_COLORS[idx % len(LANE_COLORS)]
        points = lane.points.copy()
        points[:, 0] *= img.shape[1]
        points[:, 1] *= img.shape[0]
        points = points.round().astype(int)
        for point in points:
            cv2.circle(img, tuple(point), 3, color=color, thickness=-1)
    return img


# Standard TuSimple h_samples for 720p images: y from 160 to 710, step 10
TUSIMPLE_H_SAMPLES = list(range(160, 720, 10))


def pred2tusimple(pred, h_samples, img_h, img_w):
    """Convert Lane predictions to TuSimple annotation format.

    Args:
        pred: List of Lane objects (from model.decode with as_lanes=True).
              Each Lane is callable: lane(ys) returns interpolated x coords.
        h_samples: List of y pixel coordinates to sample.
        img_h: Original image height.
        img_w: Original image width.
    Returns:
        lanes: List of lists, each inner list is x-coords at h_samples (-2 = invalid).
    """
    ys = np.array(h_samples, dtype=np.float64) / img_h  # normalize to [0,1]
    lanes = []
    for lane in pred:
        xs = lane(ys)  # spline interpolation, -2 for out-of-range
        pixel_xs = (xs * img_w).astype(int)
        pixel_xs[(xs < 0) | (xs > 1)] = -2  # mark invalid
        lanes.append(pixel_xs.tolist())
    return lanes


def parse_args():
    parser = argparse.ArgumentParser(description='LaneATT PyTorch batch inference on a directory')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing JPG images')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34'],
                        help='Backbone network (default: resnet18)')
    parser.add_argument('--anchors_freq_path', type=str, default='data/tusimple_carla_anchor_mask.pt',
                        help='Path to anchor frequency file')
    parser.add_argument('--topk_anchors', type=int, default=1000, help='Top-K anchors to keep')
    parser.add_argument('--model_path', type=str,
                        default='experiments/laneatt_r18_tusimple/backup_models/model_0043_carla0329.pt',
                        help='Path to PyTorch checkpoint (.pt)')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=50.0, help='NMS overlap threshold')
    parser.add_argument('--nms_topk', type=int, default=4, help='Max lanes to keep after NMS')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: <image_dir>/lane_results)')
    parser.add_argument('--anno', action='store_true',
                        help='Generate TuSimple annotation JSON file')
    parser.add_argument('--anno_file', type=str, default=None,
                        help='Annotation output path (default: <output_dir>/annotations.json)')
    parser.add_argument('--h_samples', type=str, default=None,
                        help='Comma-separated h_samples, or "auto" to use standard TuSimple 720p samples (default: auto)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.isdir(args.image_dir):
        print(f'Directory not found: {args.image_dir}')
        exit(1)

    # Collect jpg files
    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])
    if not image_files:
        print(f'No JPG files found in {args.image_dir}')
        exit(1)

    # Output directory
    output_dir = args.output_dir or os.path.join(args.image_dir, 'lane_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LaneATT(backbone=args.backbone,
                    anchors_freq_path=args.anchors_freq_path,
                    topk_anchors=args.topk_anchors)
    state_dict = torch.load(args.model_path, map_location=device)['model']
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Determine h_samples for annotation
    if args.h_samples:
        h_samples = [int(x) for x in args.h_samples.split(',')]
    else:
        h_samples = TUSIMPLE_H_SAMPLES  # standard 720p: range(160, 720, 10)

    anno_path = args.anno_file or os.path.join(output_dir, 'annotations.json')

    print(f'Model loaded: {args.model_path}')
    print(f'Processing {len(image_files)} images from {args.image_dir}')
    if args.anno:
        print(f'Annotation h_samples: {h_samples[0]}..{h_samples[-1]}, step={h_samples[1]-h_samples[0]}, count={len(h_samples)}')

    total = len(image_files)
    detected = 0
    anno_list = []

    with torch.no_grad():
        for idx, fname in enumerate(image_files):
            img_path = os.path.join(args.image_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print(f'[{idx+1}/{total}] {fname} -- skipped (unreadable)')
                continue

            img_h, img_w = img.shape[:2]
            img_pre = preprocess(img).to(device)
            output = model(img_pre, conf_threshold=args.conf,
                           nms_thres=args.nms_thres, nms_topk=args.nms_topk)
            pred = model.decode(output, as_lanes=True)[0]

            n_lanes = len(pred)
            if n_lanes > 0:
                detected += 1
                result_img = draw_lanes(img, pred)
            else:
                result_img = img

            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, result_img)

            # Generate TuSimple annotation
            if args.anno:
                lanes = pred2tusimple(pred, h_samples, img_h, img_w) if n_lanes > 0 else []
                anno = {
                    'raw_file': fname,
                    'h_samples': h_samples,
                    'lanes': lanes,
                }
                anno_list.append(anno)

            print(f'[{idx+1}/{total}] {fname}: {n_lanes} lanes')

    # Save annotation file (JSON lines, same as TuSimple format)
    if args.anno and anno_list:
        with open(anno_path, 'w') as f:
            for anno in anno_list:
                f.write(json.dumps(anno) + '\n')
        print(f'Annotations saved to {anno_path} ({len(anno_list)} entries)')

    print(f'\nDone. {detected}/{total} images had lanes detected.')
    print(f'Results saved to {output_dir}')
