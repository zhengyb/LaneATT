# Convert Openlane V1.x images and lane annotation to Tusimple format

import json
import os

import cv2
import numpy as np
from scipy.interpolate import splprep, splev

TUSIMPLE_IMG_RES = (720, 1280)

def get_cut_h(old_img_shape=(1280, 1920)):
    tusimple_h, tusimple_w = TUSIMPLE_IMG_RES
    img_h, img_w = old_img_shape
    resized_h = int(round((tusimple_w * img_h) / img_w))
    cut_h = (resized_h - tusimple_h)
    return cut_h

def convert_image_one(openlane_img_path, tusimple_img_path):
    # Read Openlane V1.x image
    openlane_img = cv2.imread(openlane_img_path)
    img_h, img_w = openlane_img.shape[:2]
    tusimple_h, tusimple_w = TUSIMPLE_IMG_RES
    assert img_w == 1920, 'image width is not equal to 1920'
    assert img_h == 1280, 'image height is not equal to 1280'
    resized_h = int(round((tusimple_w * img_h) / img_w))
    # resize image
    resized_img = cv2.resize(openlane_img, (tusimple_w, resized_h))
    # cut image
    cut_h = get_cut_h(old_img_shape=(img_h, img_w))
    #print(f"cut_h: {cut_h}")
    cut_img = resized_img[cut_h:, :, :]
    # save image
    cv2.imwrite(tusimple_img_path, cut_img)
    return cut_img

def interpolate_lane(points, n=50):
    """Spline interpolation of a lane. Used on the predictions"""
    x = [x for x, _ in points]
    y = [y for _, y in points]
    tck, _ = splprep([x, y], s=0, t=n, k=min(3, len(points) - 1))

    u = np.linspace(0., 1., n) # 生成50个均匀分布的点，归一化
    return np.array(splev(u, tck)).T # 插值，并转换成(n, 2)的形状

def convert_lane_one(lane_uv, old_img_shape=(1280, 1920), h_samples=list(range(160, 720, 10))):
    tusimple_h, tusimple_w = TUSIMPLE_IMG_RES
    lane_uv = np.array(lane_uv)
    img_h, img_w = old_img_shape
    resized_h = int(round((tusimple_w * img_h) / img_w))
    lane_points = []
    # format and resize lane points
    for u, v in zip(lane_uv[0], lane_uv[1]):
        lane_points.append([(u * tusimple_w / old_img_shape[1]), 
                            int(round(v * resized_h / old_img_shape[0]))])
    # remove cutted points
    cut_h = get_cut_h(old_img_shape=old_img_shape)
    #print(f"cut_h: {cut_h}")
    lane_points = [[p[0], p[1] - cut_h] for p in lane_points if p[1] > cut_h]
    # sort lane points by v value
    lane_points = sorted(lane_points, key=lambda x: x[1])
    # 按v值分组并计算u坐标平均值
    v_groups = {}
    for point in lane_points:
        v = point[1]
        if v not in v_groups:
            v_groups[v] = []
        v_groups[v].append(point[0])  # 收集相同v值的所有u坐标
    
    # 生成新的车道点列表（每个v值保留一个平均后的点）
    lane_points = [
        [sum(us) / len(us), v]  # 计算u坐标平均值
        for v, us in sorted(v_groups.items(), key=lambda x: x[0])  # 按v值排序
    ]
    #new_lane_points = [[int(p[0]), int(p[1])] for p in lane_points if p[0] >= 0]
    #return new_lane_points
    ys = range(0, 720, 1)
    xs = [-2.0] * len(ys)

    new_lane_points = [[x, y] for x, y in zip(xs, ys)]
    # fill with lane_points
    for point in lane_points:
        new_lane_points[point[1]] = [point[0], point[1]]
    y_start = lane_points[0][1]
    y_end = lane_points[-1][1]
    last_point = None
    next_point = None

    #print(f"y_start: {y_start}, y_end: {y_end}")
    # fill the gap between y_start and y_end
    for y in range(y_start, y_end, 1):
        if new_lane_points[y][0] >= 0:
            last_point = new_lane_points[y]
            next_point = None
            continue

        # A gap point is found at y
        if next_point is None:
            # find the next non-negative x point
            for y2 in range(y+1, y_end+1, 1):
                if new_lane_points[y2][0] >= 0:
                    next_point = new_lane_points[y2]
                    break
        # interpolate between y and y2
        # print(f"last_point: {last_point}, next_point: {next_point}")
        if last_point is not None and next_point is not None:
            new_x = last_point[0] + (next_point[0] - last_point[0]) * (y - last_point[1]) / (next_point[1] - last_point[1])
            new_lane_points[y] = [new_x, y]
            #print(f"interpolate ({new_x},{y}) between ({last_point[0]},{last_point[1]}) and ({next_point[0]},{next_point[1]})")
        else:
            print(f"no next point found for {y}")
            raise ValueError(f"no next point found for {y}")

    new_lane_points = [[int(round(p[0])), int(round(p[1]))] for p in new_lane_points]
    #new_lane_points = np.array(new_lane_points)
    tusimple_lane = [new_lane_points[h_samples[i]][0] for i in range(len(h_samples)) ]
    return tusimple_lane

def convert_label_one(openlane_label_path, tusimple_label_path, openlane_img_path, tusimple_img_path, h_samples=list(range(160, 720, 10))):
    img_h, img_w = cv2.imread(openlane_img_path).shape[:2]
    tusimple_label = {
        'lanes': [],
        'old_lanes': [],
        'h_samples': [],
        'raw_file': tusimple_img_path
    }
    # Read Openlane V1.x lane annotation
    with open(openlane_label_path, 'r') as f:
        old_label = json.load(f)

    old_lanes = [lane['uv'] for lane in old_label['lane_lines'] if lane['attribute'] > 0 and lane['attribute'] < 5]
    tusimple_label['old_lanes'] = old_lanes
    tusimple_lanes = [convert_lane_one(lane) for lane in old_lanes]
    tusimple_label['lanes'] = tusimple_lanes
    tusimple_label['h_samples'] = h_samples

    with open(tusimple_label_path, 'w') as f:
        json.dump(tusimple_label, f)
    return tusimple_label


def draw_image(new_img, new_label, output_path):
    h_samples = new_label['h_samples']
    for lane in new_label['lanes']:
        for pi in range(len(lane)):
            x = lane[pi]
            y = h_samples[pi]
            cv2.circle(new_img, (x, y), 1, (0, 0, 255), -1)
    cv2.imwrite(output_path, new_img)

if __name__ == '__main__':

    old_path = './datasets/openlane/images/training/segment-15832924468527961_1564_160_1584_160_with_camera_labels'
    old_label_path = './datasets/openlane/training/segment-15832924468527961_1564_160_1584_160_with_camera_labels'
    new_path = './'
    #filename = '150767882687643500.jpg'
    filename = '150767882727675700.jpg'
    label_filename = filename.replace('.jpg', '.json')
    ret_filename = filename.replace('.jpg', '_ret.jpg')

    # convert image
    new_img = convert_image_one(os.path.join(old_path, filename), os.path.join(new_path, filename))
    # convert label
    new_label = convert_label_one(os.path.join(old_label_path, label_filename), os.path.join(new_path, label_filename), 
                      os.path.join(old_path, filename), os.path.join(new_path, filename))
    # draw image
    draw_image(new_img, new_label, os.path.join(new_path, ret_filename))