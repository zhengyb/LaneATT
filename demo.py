import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline
from imgaug.augmentables.lines import LineString, LineStringsOnImage
import os

from lib.config import Config
from lib.experiment import Experiment
from lib.datasets.lane_dataset import LaneDataset


PRED_HIT_COLOR = (0, 255, 0)  # green

LANE_COLORS = [
    (0, 0, 255),  # red
    (0, 255, 0),  # green
    (255, 0, 0),  # blue
    (0, 255, 255),  # yellow
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (255, 255, 255),  # white
    (128, 128, 128),  # gray
    (128, 0, 0),  # maroon
    (128, 128, 0),  # olive
    (0, 128, 128),  # teal
    (0, 0, 128),  # navy
]


def parse_args():
    parser = argparse.ArgumentParser(description="Video Lane Detection Demo")
    parser.add_argument(
        "--video", type=str, default="./datasets/route28.mp4", help="Input video path"
    )
    parser.add_argument(
        "--output", type=str, default="./output-video.mp4", help="Output video path"
    )
    parser.add_argument("--output_fps", type=int, default=2, help="Output video FPS")

    return parser.parse_args()


def get_metrics(lanes, _):
    return 0, 0, [1] * len(lanes), [1] * len(lanes)


def draw_annotation(pred=None, img=None):
    img_h, _, _ = img.shape
    data = []
    if pred is not None:
        # print(len(pred), 'preds')
        fp, fn, matches, accs = get_metrics(pred, None)
        assert len(matches) == len(pred)
        data.append((matches, accs, pred))
    else:
        fp = fn = None

    if True:
        for i, lane in enumerate(pred):
            color = LANE_COLORS[i % len(LANE_COLORS)]
            points = lane.points  # 归一化坐标
            points[:, 0] *= img.shape[1]  # 转换为像素坐标
            points[:, 1] *= img.shape[0]
            points = points.round().astype(int)  # 取整
            xs, ys = points[:, 0], points[:, 1]
            # 用相邻点连线绘制2D车道线
            for curr_p, next_p in zip(points[:-1], points[1:]):
                img = cv2.line(
                    img,
                    tuple(curr_p),
                    tuple(next_p),
                    color=color,
                    thickness=3 if matches is None else 3,
                )

    return img, fp, fn


def crop_image(img, crop_ratio=0.2):
    img_h, img_w, _ = img.shape
    half_crop_ratio = crop_ratio / 2

    img_ret = img[
        int(img_h * crop_ratio) :,
        int(img_w * half_crop_ratio) : int(img_w * (1 - half_crop_ratio)),
    ]
    return img_ret


def process_image_path(model, img_path, img_size, device, test_parameters):
    img = cv2.imread(img_path)
    #img = crop_image(img)
    img, fp, fn, prediction = process_one_frame(model, img, img_size, device, test_parameters)
    result_path = img_path.replace('/', '_').replace('.jpg', '_result.jpg')
    cv2.imwrite(result_path, img)
    return img, fp, fn, prediction

def process_one_frame(model, img, img_size, device, test_parameters):
    img = cv2.resize(img, (img_size[1], img_size[0]))
    img_org = img.copy()
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # ?
    img = img / 255.0  # 归一化

    # float32
    img = img.astype(np.float32)
    # 转换为Tensor
    img_tensor = ToTensor()(img)
    # 添加batch维度
    images = img_tensor.unsqueeze(0)
    # 添加到设备
    images = images.to(device)
    # 模型推理
    output = model(images, **test_parameters)
    prediction = model.decode(output, as_lanes=True)

    # 后处理
    img, fp, fn = draw_annotation(prediction[0], img_org)

    return img, fp, fn, prediction[0]


def main():
    args = parse_args()
    exp = Experiment("laneatt_r18_demo", args, mode="test")
    cfg_path = exp.cfg_path
    # 1. 加载配置和模型
    cfg = Config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 初始化模型
    model = cfg.get_model()
    epoch = exp.get_last_checkpoint_epoch()
    model_path = exp.get_checkpoint_path(epoch)
    print("Loading model %s", model_path)
    model.load_state_dict(exp.get_epoch_model(epoch))
    model = model.to(device)
    # model.eval()

    # 3. 预处理参数

    img_size = cfg["datasets"]["test"]["parameters"]["img_size"]
    normalize = cfg["datasets"]["test"]["parameters"]["normalize"]
    test_parameters = cfg.get_test_parameters()

    # 4. 初始化视频流
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / args.output_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("height: ", height, "width: ", width, "total_frames: ", total_frames, "fps: ", fps, "frame_interval: ", frame_interval)

    # 5. 准备输出视频
    if os.path.exists(args.output):
        os.remove(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        args.output, fourcc, args.output_fps, (img_size[1], img_size[0])
    )

    #skip_frames = 30 * 60 * 2
    skip_frames = 0
    handle_frames = 2 * 60 * 10
    stop_frames = skip_frames + (frame_interval * handle_frames)
    stop_frames = min(stop_frames, total_frames)
    handle_frames = int((stop_frames - skip_frames) / frame_interval)
    print("handle_frames: ", handle_frames)
    print("stop_frames: ", stop_frames)
    print("total_frames: ", total_frames)

    output_dir = "./outputs"
    os.system(f"rm -rf {output_dir}/*")
    os.makedirs(output_dir, exist_ok=True)

    process_image_path(model, "/app/datasets/tusimple_test_image/0.jpg", img_size, device, test_parameters)
    process_image_path(model, "/app/datasets/tusimple_test_image/1.jpg", img_size, device, test_parameters)
    process_image_path(model, "/app/datasets/tusimple_test_image/2.jpg", img_size, device, test_parameters)

    frame_count = 0
    with torch.no_grad():
        progress = tqdm(total=handle_frames, desc="Processing video")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count < skip_frames:
                frame_count += 1
                continue
            if frame_count % frame_interval != 0:
                frame_count += 1
                continue
            # 预处理帧
            img = frame
            #img = crop_image(img)
            #img = cv2.rotate(img, cv2.ROTATE_180)

            img, fp, fn, prediction = process_one_frame(
                model, img, img_size, device, test_parameters
            )

            # 写入/显示结果
            out.write(img)
            output_path = os.path.join(output_dir, f"image_{frame_count}.jpg")
            cv2.imwrite(output_path, img)
            progress.update(1)

            frame_count += 1
            if frame_count > stop_frames:
                break

    print("frame_count: ", frame_count)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing completed. Output saved to {args.output}")


if __name__ == "__main__":
    main()
