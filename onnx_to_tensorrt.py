import os
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

from nms import nms
import common

TRT_LOGGER = trt.Logger()


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

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser, trt.Runtime(TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run laneatt_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 360, 640]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            print(f"Engine saved to {engine_file_path}")
            return engine

    if os.path.exists(engine_file_path):
        # FIXME: getPluginCreator could not find plugin: ScatterND version: 1
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def do_nms(proposals, conf_threshold=0.4, nms_thres=50., nms_topk=4):
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
    start_time = time.perf_counter()
    
    # proposals_to_pred
    n_strips = n_offsets - 1
    anchor_ys = torch.linspace(1, 0, steps=n_offsets, dtype=torch.float32, device='cuda:0')
    anchor_ys = anchor_ys.double()
    lanes = []
    #print(f"proposals length: {len(proposals)}")
    for lane in proposals:
        lane_xs = lane[5:] / 640
        start = int(round(lane[2].item() * n_strips))
        length = int(round(lane[4].item()))
        end = start + length - 1
        end = min(end, len(anchor_ys) - 1)
        # end = label_end
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
    # print('Number of lanes: {}'.format(len(lanes)))

    # Visualize
    img_h, img_w = img.shape[:2]
    #print(f"lanes length: {len(lanes)}")
    for idx, lane_points in enumerate(lanes):
        # Get color using modulo to cycle through predefined colors
        color = PREDEFINED_COLORS[idx % len(PREDEFINED_COLORS)]
        
        # scale back to input size
        lane_points[:, 0] *= img_w
        lane_points[:, 1] *= img_h
        lane_points = lane_points.round().astype(int)
        for point in lane_points:
            cv2.circle(img, tuple(point), 2, color, -1)
    #cv2.imshow('LaneATT_tensorrt', img)
    #cv2.waitKey(0)
    # Save the image with the result
    if image_file_path is not None:
        output_file_path = image_file_path.split('.jpg')[0] + '_result.jpg'
        cv2.imwrite(output_file_path, img)
        print(f"Result saved to {output_file_path}")

    # 添加计时输出
    elapsed = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
    # print(f"后处理耗时: {elapsed:.2f}ms")
    return img


def engine_inference_video(onnx_file_path, video_file_path, target_fps=5.0):
    """Create a TensorRT engine for ONNX-based LaneATT and run inference on a video."""
    engine_file_path = onnx_file_path.split('.onnx')[0] + '.trt8'

    # Load the video
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / target_fps)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file_path}")
        return
    
    # Create output video writer
    output_video_path = video_file_path.split('.mp4')[0] + '_result.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outv = cv2.VideoWriter(output_video_path, fourcc, target_fps, (1280, 720))
    
    # Get the original fps of the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    
    ZYB_VIDEO = False
    if video_file_path.find("route28") != -1:
        ZYB_VIDEO = True
    if ZYB_VIDEO:
        skip_frames = fps * 60 * 1
    else:
        skip_frames = 0
    handle_frames = target_fps * 60 * 20
    stop_frames = skip_frames + (frame_interval * handle_frames)
    stop_frames = min(stop_frames, total_frames)
    handle_frames = int((stop_frames - skip_frames) / frame_interval)
    print("handle_frames: ", handle_frames)
    print("stop_frames: ", stop_frames)
    print("total_frames: ", total_frames)

    os.system('rm -rf ./datasets/route28_result/*')
    os.makedirs('./datasets/route28_result', exist_ok=True)
    frame_count = 0
    # Do inference with TensorRT
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count < skip_frames:
                frame_count += 1
                continue
            if frame_count % frame_interval == 0:
                print(f"Processing frame {frame_count}...")
                # Load test image
                if ZYB_VIDEO:
                    image_raw = cv2.rotate(frame, cv2.ROTATE_180)
                else:
                    image_raw = frame
                image_raw = cv2.resize(image_raw, (1280, 720), cv2.INTER_LINEAR)
                #image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                image = image_raw.copy()
                image = cv2.resize(image, (640, 360), cv2.INTER_LINEAR)
                # normalize and flatten
                image = image.astype(np.float32) / 255.0
                image = image.transpose([2, 0, 1]).flatten() # (H, W, C) -> (C, H, W) -> (C * H * W,)

                # Do inference with TensorRT
                #with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
                if True:                    
                    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
                    inputs[0].host = image
                    trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

                # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
                output = trt_outputs[0].reshape([1000, 77])

                proposals = do_nms(output, conf_threshold=0.5, nms_thres=50., nms_topk=5)
                result_img = post_process(image_raw, proposals, image_file_path=None)   
                cv2.imwrite(f'./datasets/route28_result/image_{frame_count}.jpg', image_raw)
                cv2.imwrite(f'./datasets/route28_result/result_{frame_count}.jpg', result_img)
                outv.write(result_img)
                #print(f"Frame {frame_count} processed")
            frame_count += 1
            if frame_count > stop_frames:
                break
    outv.release()
    cap.release()
    print(f"Output video saved to {output_video_path}")



def engine_inference(onnx_file_path, image_file_path, benchmark=False, 
                    warmup=10, repeats=100, profile_mem=False):
    """Create a TensorRT engine for ONNX-based LaneATT and run inference.
    
    Args:
        benchmark: 是否进行推理速度测试 (默认False)
        warmup: 预热迭代次数 (默认10)
        repeats: 正式测试迭代次数 (默认100)
    """
    engine_file_path = onnx_file_path.split('.onnx')[0] + '.trt8'

    # Load test image
    image_raw = cv2.imread(image_file_path)
    image = cv2.resize(image_raw, (640, 360), cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image.transpose([2, 0, 1]).flatten() # (H, W, C) -> (C, H, W) -> (C * H * W,)

    # Do inference with TensorRT
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        print('\nRunning inference on image {}...'.format(image_file_path))
        inputs[0].host = image
        
        if benchmark:
            # 初始化测量工具
            start_gpu = cuda.Event()
            end_gpu = cuda.Event()
            mem_tracker = []

            # 预热阶段
            for _ in range(warmup):
                common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            
            # 内存分析
            if profile_mem:
                free_start, _ = cuda.mem_get_info()
                torch.cuda.reset_peak_memory_stats()

            # 时间测量
            stream.synchronize()
            start_cpu = time.perf_counter()
            
            # 新版PyCUDA事件记录方式
            start_gpu.record(stream)
            for _ in range(repeats):
                common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            end_gpu.record(stream)
            
            stream.synchronize()
            elapsed_cpu = (time.perf_counter() - start_cpu) * 1000 / repeats
            # 新版时间差计算方式
            elapsed_gpu = start_gpu.time_till(end_gpu) / repeats

            # 输出结果
            print(f"┌{'─'*40}┐")
            print(f"│ 推理速度分析 (平均{repeats}次)    │")
            print(f"├{'─'*40}┤")
            print(f"│ CPU端耗时: {elapsed_cpu:>8.2f} ms/帧   │")
            print(f"│ GPU端耗时: {elapsed_gpu:>8.2f} ms/帧   │")
            print(f"│ 理论FPS  : {1000/elapsed_gpu:>8.1f} FPS      │")

            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            if profile_mem:
                used_mem = (free_start - min(mem_tracker)) / 1024**2
                print(f"├{'─'*40}┤")
                print(f"│ 峰值显存使用: {used_mem:.2f} MB       │")
                print(f"└{'─'*40}┘")
        else:
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    if not benchmark:
        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
        output = trt_outputs[0].reshape([1000, 77])

        proposals = do_nms(output, conf_threshold=0.5, nms_thres=50., nms_topk=5)
        post_process(image_raw, proposals, image_file_path=image_file_path)
    else:
        # 添加后处理基准测试
        post_times = []
        for _ in range(repeats):
            output = trt_outputs[0].reshape([1000, 77])
            proposals = do_nms(output, conf_threshold=0.5, nms_thres=50., nms_topk=5)
            start_post = time.perf_counter()
            _ = post_process(image_raw.copy(), proposals)  # 使用拷贝避免修改原图
            post_times.append((time.perf_counter() - start_post) * 1000)
        
        post_process(image_raw, proposals, image_file_path=image_file_path)
        avg_post = sum(post_times)/len(post_times)
        print(f"├{'─'*40}┤")
        print(f"│ 后处理耗时: {avg_post:>8.2f} ms/帧   │")
        print(f"│ 端到端FPS : {1000/(elapsed_gpu + avg_post):>8.1f} FPS      │")
        print(f"└{'─'*40}┘")


if __name__ == '__main__':
    #image_file = './samples/02610.jpg'
    image_file = './datasets/tusimple_test_image/2.jpg'
    #image_file = './datasets/route28_result/5625.jpg'
    onnx_file = './LaneATT_r18_llamas.onnx'
    engine_inference(onnx_file, image_file, benchmark=True, warmup=10, repeats=100)
    #engine_inference_video(onnx_file, './datasets/route28.mp4')
    # Speed test on RTX4090 PC:
    # ┌────────────────────────────────────────┐
    # │ 推理速度分析 (平均100次)                  │
    # ├────────────────────────────────────────┤
    # │ CPU端耗时:     0.97 ms/帧               │
    # │ GPU端耗时:     0.97 ms/帧               │
    # │ 理论FPS  :   1031.9 FPS                 │
    # ├────────────────────────────────────────┤
    # │ 后处理耗时:     1.06 ms/帧               │
    # │ 端到端FPS :    491.8 FPS                │
    # └────────────────────────────────────────┘