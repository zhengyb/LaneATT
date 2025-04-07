import cv2
import torch
import numpy as np
import onnxruntime as ort

class LaneATT(object):
    def __init__(self, model_path, S=72, img_w=640, img_h=360, conf_thresh=0.5, nms_thres=50., nms_topk=4) -> None:
        self.predictor   = ort.InferenceSession(model_path, provider_options=["CPUExecutionProvider"])
        self.n_strips    = S - 1
        self.n_offsets   = S
        self.img_w       = img_w
        self.img_h       = img_h
        self.conf_thresh = conf_thresh
        self.nms_thres   = nms_thres
        self.nms_topk    = nms_topk
        self.anchor_ys   = [1 - i / self.n_strips for i in range(self.n_offsets)]

    def preprocess(self, img):
        # 1. resize
        img = cv2.resize(img, (self.img_w, self.img_h))
        # 2. normalize
        img = (img / 255.0).astype(np.float32)
        # 3. to bchw
        img = img.transpose(2, 0, 1)[None]
        return img
    
    def forward(self, input):
        # input->1x3x360x640
        output = self.predictor.run(None, {"images": input})[0]
        return output

    def postprocess(self, pred):
        # pred->1x1000x77
        lanes = []
        for img_id, lane_id in zip(*np.where(pred[..., 1] > self.conf_thresh)):
            lane = pred[img_id, lane_id]
            lanes.append(lane.tolist())
        lanes = sorted(lanes, key=lambda x:x[1], reverse=True)
        lanes = self._nms(lanes)
        lanes_points = self._decode(lanes)
        return lanes_points[:self.nms_topk]

    def _nms(self, lanes):
        
        remove_flags = [False] * len(lanes)
        
        keep_lanes = []
        for i, ilane in enumerate(lanes):
            if remove_flags[i]:
                continue
                
            keep_lanes.append(ilane)
            for j in range(i + 1, len(lanes)):
                if remove_flags[j]:
                    continue
                
                jlane = lanes[j]
                if self._lane_iou(ilane, jlane) < self.nms_thres:
                    remove_flags[j] = True
        return keep_lanes
    
    def _lane_iou(self, lane_a, lane_b):
        # lane = (_, conf, start_y, start_x, length, ...) = 2+2+1+72 = 77
        start_a = int(lane_a[2] * self.n_strips + 0.5)
        start_b = int(lane_b[2] * self.n_strips + 0.5)
        start   = max(start_a, start_b)
        end_a   = start_a + int(lane_a[4] + 0.5) - 1
        end_b   = start_b + int(lane_b[4] + 0.5) - 1
        end     = min(min(end_a, end_b), self.n_strips)
        dist = 0
        for i in range(start, end + 1):
            dist += abs(lane_a[i + 5] - lane_b[i + 5])
        dist = dist / float(end - start + 1)
        return dist

    def _decode(self, lanes):
        lanes_points = []
        for lane in lanes:
            start  = int(lane[2] * self.n_strips + 0.5)
            end    = start + int(lane[4] + 0.5) - 1
            end    = min(end, self.n_strips)
            points = []
            for i in range(start, end + 1):
                points.append([lane[i + 5] / self.img_w, self.anchor_ys[i]])
            points = torch.from_numpy(np.array(points))
            lanes_points.append(points)
        return lanes_points

if __name__ == "__main__":
    image_file = './datasets/tusimple_test_image/2.jpg'
    image = cv2.imread(image_file)

    print(f"Processing image: {image_file}")

    model_file_path = "laneatt.sim.onnx"
    model   = LaneATT(model_file_path)
    img_pre = model.preprocess(image)
    pred    = model.forward(img_pre)
    lanes_points = model.postprocess(pred)

    for points in lanes_points:
        points[:, 0] *= image.shape[1]
        points[:, 1] *= image.shape[0]
        points = points.numpy().round().astype(int)
        # for curr_p, next_p in zip(points[:-1], points[1:]):
        #     cv2.line(image, tuple(curr_p), tuple(next_p), color=(0, 255, 0), thickness=3)
        for point in points:
            cv2.circle(image, point, 3, color=(0, 255, 0), thickness=-1)
    
    cv2.imwrite("result.jpg", image)
    print(f"Result saved to result.jpg")