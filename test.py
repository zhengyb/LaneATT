import cv2
import torch
import numpy as np
from lib.models.laneatt import LaneATT

def preprocess(img, dst_width=640, dst_height=360):
    img_pre = cv2.resize(img, (dst_width, dst_height))
    img_pre = (img_pre / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    img = cv2.imread("datasets/tusimple_test_image/0.jpg")
    img_pre = preprocess(img).to(device)

    #model = LaneATT(anchors_freq_path="data/culane_anchors_freq.pt", topk_anchors=1000)
    #state_dict = torch.load("experiments/laneatt_r34_culane/models/model_0015.pt")['model']
    model = LaneATT(backbone='resnet18', anchors_freq_path="data/tusimple_anchors_freq.pt", topk_anchors=1000)
    state_dict = torch.load("experiments/laneatt_r18_tusimple/models/model_0100.pt")['model']
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_pre, conf_threshold=0.5, nms_thres=50.0, nms_topk=4)
        pred = model.decode(output, as_lanes=True)[0]
        for line in pred:
            points = line.points
            points[:, 0] *= img.shape[1]
            points[:, 1] *= img.shape[0]
            points = points.round().astype(int)
            for point in points:
                cv2.circle(img, point, 3, color=(0, 255, 0), thickness=-1)
        cv2.imwrite("result.jpg", img)
