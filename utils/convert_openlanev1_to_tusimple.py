# Convert Openlane V1.x images and lane annotation to Tusimple format

import json
import os

import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import shutil
import random

TUSIMPLE_IMG_RES = (720, 1280)

LANE_COLORS = [
    (0, 0, 255), # yellow
    (0, 255, 0), # green
    (255, 0, 0), # red
    (0, 255, 255), # cyan
    (255, 255, 0), # yellow
    (255, 0, 255), # magenta
    (0, 0, 128), # blue
    (0, 128, 0), # green
    (128, 0, 0), # red
    (0, 128, 128), # cyan
    (128, 128, 0), # yellow
    (128, 0, 128), # magenta
    (128, 128, 128), # gray
    (128, 128, 128), # gray
    (128, 128, 128), # gray
    (128, 128, 128), # gray
    (128, 128, 128), # gray
    (128, 128, 128), # gray
    (128, 128, 128), # gray    
]

def get_cut_h(old_img_shape=(1280, 1920)):
    tusimple_h, tusimple_w = TUSIMPLE_IMG_RES
    img_h, img_w = old_img_shape
    resized_h = int(round((tusimple_w * img_h) / img_w))
    cut_h = (resized_h - tusimple_h)
    return cut_h

def convert_image_one(openlane_img_path, tusimple_img_path):
    """
    Convert Openlane V1.x image to Tusimple image
    """
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
    if len(lane_points) < 2:
        print(f"invalid lane_points: {len(lane_points)}")
        return []
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
    valid_tusimple_lane = [x for x in tusimple_lane if x >= 0]
    if len(valid_tusimple_lane) < 2:
        print(f"invalid tusimple lane: {len(valid_tusimple_lane)}")
        return []
    return tusimple_lane

def convert_label_one(openlane_label_path, openlane_img_path, tusimple_img_path, tusimple_label_path=None, 
                      h_samples=list(range(160, 720, 10)), with_old_lane=False, only_4_lanes=True):
    """
    Convert Openlane V1.x label to Tusimple label
    """
    img_h, img_w = cv2.imread(openlane_img_path).shape[:2]
    raw_file = tusimple_img_path.split('/')[3:]
    raw_file = '/'.join(raw_file)
    tusimple_label = {
        'lanes': [],
        'h_samples': [],
        'raw_file': raw_file
    }
    # Read Openlane V1.x lane annotation
    with open(openlane_label_path, 'r') as f:
        old_label = json.load(f)

    if only_4_lanes:
        old_lanes = [lane['uv'] for lane in old_label['lane_lines'] if lane['attribute'] > 0 and lane['attribute'] < 5]
    else:
        old_lanes = [lane['uv'] for lane in old_label['lane_lines']]
    if with_old_lane:
        tusimple_label['old_lanes'] = old_lanes
    tusimple_lanes = []
    for lane in old_lanes:
        new_lane = convert_lane_one(lane)
        if len(new_lane) > 0:
            tusimple_lanes.append(new_lane)
    tusimple_label['lanes'] = tusimple_lanes
    tusimple_label['h_samples'] = h_samples

    if tusimple_label_path is not None:
        with open(tusimple_label_path, 'w') as f:
            json.dump(tusimple_label, f)
    return tusimple_label


def convert_dir(old_img_dir, old_label_dir, new_img_dir_parent_path, new_label_dir_parent_path, split_name='valid', 
                sample_interval=5, convert_img=True, force_convert_img=False):
    this_dir_name = old_img_dir.split('/')[-1]
    new_img_dir_path = os.path.join(new_img_dir_parent_path, 'ol_' + this_dir_name)
    os.makedirs(new_img_dir_path, exist_ok=True)
    os.makedirs(new_label_dir_parent_path, exist_ok=True)
    new_label_filepath = os.path.join(new_label_dir_parent_path, 'ol_' + split_name + '_' + this_dir_name+'.json')
    
    print(f"Processing {old_img_dir}")
    image_cnt = 0
    with open(new_label_filepath, 'w') as f:
        for label_filename in os.listdir(old_label_dir):
            if label_filename.endswith('.json'):
                if image_cnt % sample_interval != 0:
                    image_cnt += 1
                    continue
                #print(f"Processing {label_filename}")
                old_img_path = os.path.join(old_img_dir, label_filename.replace('.json', '.jpg'))
                old_label_path = os.path.join(old_label_dir, label_filename)
                new_img_path = os.path.join(new_img_dir_path, label_filename.replace('.json', '.jpg'))
                if convert_img:
                    if (not os.path.exists(new_img_path)) or force_convert_img:
                        convert_image_one(old_img_path, new_img_path)
                label = convert_label_one(old_label_path, old_img_path, new_img_path)
                json.dump(label, f)
                f.write('\n')
                image_cnt += 1
    print(f"Processed {image_cnt}/{sample_interval} images")
    return new_label_filepath

def get_tusimple_label_filename(subdir, split_name):
    return 'ol_' + split_name + '_' + subdir + '.json'

def convert_dataset_dir(split, old_dataset_label_dir, old_dataset_img_dir, new_dataset_dir, 
                        sample_interval=5, convert_img=True):
    first_subdir = None
    first_tusimple_label_filename = None
    new_img_base_dir = os.path.join(new_dataset_dir, 'clips', 'openlane')
    for subdir in os.listdir(old_dataset_label_dir):
        if os.path.isdir(os.path.join(old_dataset_label_dir, subdir)):
            old_img_dir = os.path.join(old_dataset_img_dir, subdir)
            old_label_dir = os.path.join(old_dataset_label_dir, subdir)
            label_filepath = convert_dir(old_img_dir, old_label_dir, new_img_base_dir, 
                                         new_dataset_dir, split_name=split, sample_interval=sample_interval, 
                                         convert_img=convert_img) 
            if first_subdir is None:
                first_subdir = subdir
                first_tusimple_label_filename = label_filepath
    return first_subdir, first_tusimple_label_filename

def draw_image(new_img, new_label, output_path):
    h_samples = new_label['h_samples']
    for lane_idx, lane in enumerate(new_label['lanes']):
        for pi in range(len(lane)):
            x = lane[pi]
            y = h_samples[pi]
            color = LANE_COLORS[lane_idx]
            cv2.circle(new_img, (x, y), 1, color, -1)
    cv2.imwrite(output_path, new_img)

def draw_tusimple_label(label_filepath, image_root, output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    with open(label_filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = json.loads(line)
            img_path = os.path.join(image_root, label['raw_file'])
            out_img_path = os.path.join(output_path, label['raw_file'])
            out_img_dir = os.path.dirname(out_img_path)
            os.makedirs(out_img_dir, exist_ok=True)
            img = cv2.imread(img_path)
            draw_image(img, label, out_img_path)
            
def test_convert_one_label():
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
    new_label = convert_label_one(os.path.join(old_label_path, label_filename),                                    
                      os.path.join(old_path, filename), 
                      os.path.join(new_path, filename),
                      os.path.join(new_path, label_filename),)
    # draw image
    draw_image(new_img, new_label, os.path.join(new_path, ret_filename))


def test_convert_dir():
    old_img_dir = 'datasets/openlane/images/validation/segment-3039251927598134881_1240_610_1260_610_with_camera_labels'
    old_label_dir = 'datasets/openlane/validation/segment-3039251927598134881_1240_610_1260_610_with_camera_labels'
    new_img_dir_parent_path = 'datasets/TUSimple/tusimple/clips/openlane/'
    new_label_dir_parent_path = 'datasets/TUSimple/tusimple/'
    new_label_filepath = convert_dir(old_img_dir, old_label_dir, new_img_dir_parent_path, new_label_dir_parent_path, split_name='valid', convert_img=True)
    draw_tusimple_label(new_label_filepath, 'datasets/TUSimple/tusimple', 'outputs/')


def test_convert_dataset_dir(split='validation', sample_interval=5):
    old_dataset_label_dir = 'datasets/openlane/%s' % split
    old_dataset_img_dir = 'datasets/openlane/images/%s' % split
    if split == 'training' or split == 'validation':
        new_dataset_dir = 'datasets/TUSimple/tusimple'
    else:
        raise ValueError(f"invalid split: {split}")
    first_subdir, new_label_filepath = convert_dataset_dir(split=split, old_dataset_label_dir=old_dataset_label_dir, 
                                                           old_dataset_img_dir=old_dataset_img_dir, 
                                                           new_dataset_dir=new_dataset_dir,
                                                           sample_interval=sample_interval,
                                                           convert_img=True)
    print(f"first_subdir: {first_subdir}")
    draw_tusimple_label(new_label_filepath, 'datasets/TUSimple/tusimple', 'outputs/')


def split_validation_dataset(split_rate=0.5):
    old_label_dir = 'datasets/TUSimple/tusimple'
    new_label_dir = 'datasets/TUSimple/tusimple-test'
    for filename in os.listdir(old_label_dir):
        if os.path.isfile(os.path.join(old_label_dir, filename)) \
            and filename.endswith('.json') \
            and filename.startswith('ol_validation'):
            to_split = True if random.random() >= split_rate else False
            if to_split:
                old_label_filepath = os.path.join(old_label_dir, filename)
                new_label_filepath = os.path.join(new_label_dir, filename.replace('ol_validation_', 'ol_test_'))
                shutil.copy(old_label_filepath, new_label_filepath)
                # remove old label
                os.remove(old_label_filepath)

def convert_openlanev1_test_dataset(test_dir, output_dir, convert_img=True, force_convert_img=False):
    # test_dir: datasets/openlane/test
    # output_img_dir: datasets/TUSimple/tusimple-test/images
    # output_label_dir: datasets/TUSimple/tusimple-test/labels

    img_dir_root = test_dir.replace('/test', '/images/')
    print(f"img_dir_root: {img_dir_root}")

    output_img_dir = os.path.join(output_dir, 'clips', 'openlane')

    # 遍历test_dir中的所有子目录
    for node in os.listdir(test_dir):
        node_path = os.path.join(test_dir, node)
        if os.path.isfile(node_path) and node.endswith('.txt'):
            # scene
            scene_id = node.split('.')[0].split('_')[1:]
            scene_id.append("case")
            scene_anno_dir = "_".join(scene_id)
            tusimple_scene_anno_path = os.path.join(output_dir, 'ol_scene_' + scene_anno_dir + '.json')
            scene_cnt = 0
            print(f"Processing {scene_anno_dir}...")
            with open(tusimple_scene_anno_path, 'w') as fo:
                with open(node_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if len(line) == 0:
                            continue
                        assert line.startswith('validation/')
                        assert line.endswith('.jpg')
                        old_img_path = os.path.join(img_dir_root, line)
                        #print(f"old_img_path: {old_img_path}")
                        old_anno_path = line.replace('validation/', f'{scene_anno_dir}/').replace('.jpg', '.json')
                        old_anno_path = os.path.join(test_dir, old_anno_path)
                        #print(f"old_anno_path: {old_anno_path}")
                        assert os.path.exists(old_anno_path)
                        assert os.path.exists(old_img_path)
                        
                        new_img_path = os.path.join(output_img_dir, line.replace('validation/segment-', 'ol_segment-'))
                        if convert_img:
                            if (not os.path.exists(new_img_path)) or force_convert_img:
                                convert_image_one(old_img_path, new_img_path)
                        assert os.path.exists(new_img_path)
                        label = convert_label_one(old_anno_path, old_img_path, new_img_path)
                        json.dump(label, fo)
                        fo.write('\n')
                        scene_cnt += 1
            print(f"Done. {scene_cnt} annotations processed.")
                    

if __name__ == '__main__':
    #test_convert_one_label()
    #test_convert_dir()
    #test_convert_dataset_dir(split='validation', sample_interval=2)
    #test_convert_dataset_dir(split='training', sample_interval=2)
    #split_validation_dataset(split_rate=0.5)
    convert_openlanev1_test_dataset('datasets/openlane/test', 'datasets/TUSimple/tusimple-test')