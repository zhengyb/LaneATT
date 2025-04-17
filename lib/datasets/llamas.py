import os
import pickle as pkl
import json
import numpy as np
from tqdm import tqdm
import cv2


from .lane_dataset_loader import LaneDatasetLoader

TRAIN_LABELS_DIR = 'labels/train'
TEST_LABELS_DIR = 'labels/valid'
TEST_IMGS_DIR = 'color_images/test'
SPLIT_DIRECTORIES = {'train': 'labels/train', 'val': 'labels/valid'}
from utils.llamas_utils import get_horizontal_values_for_four_lanes
import utils.llamas_metric as llamas_metric


class LLAMAS(LaneDatasetLoader):
    def __init__(self, split='train', max_lanes=None, root=None):
        self.split = split
        self.root = root
        if split != 'test' and split not in SPLIT_DIRECTORIES.keys():
            raise Exception('Split `{}` does not exist.'.format(split))
        if split != 'test':
            self.labels_dir = os.path.join(self.root, SPLIT_DIRECTORIES[split])

        self.img_w, self.img_h = 1276, 717
        self.annotations = []
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

    # Public method
    def get_img_heigth(self, _):
        return self.img_h

    # Public method
    def get_img_width(self, _):
        return self.img_w

    # Public method
    def get_metrics(self, lanes, _):
        # Placeholders
        return [0] * len(lanes), [0] * len(lanes), [1] * len(lanes), [1] * len(lanes)

    def get_img_path(self, json_path):
        # /foo/bar/test/folder/image_label.ext --> test/folder/image_label.ext
        base_name = '/'.join(json_path.split('/')[-3:])
        image_path = os.path.join('color_images', base_name.replace('.json', '_color_rect.png'))
        return image_path

    def get_json_paths(self):
        json_paths = []
        for root, _, files in os.walk(self.labels_dir):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))
        return json_paths

    def get_json_paths_per_dir(self):
        json_paths = []
        for subdir in os.listdir(self.labels_dir):
            subdir_path = os.path.join(self.labels_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            this_dir_json_paths = {}
            this_dir_json_paths['dirname'] = subdir
            this_dir_json_paths['json_paths'] = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if file.endswith(".json")]
            json_paths.append(this_dir_json_paths)

        return json_paths


    def convert_lanes_to_tusimple_format(self, lanes):
        """
        lanes: list of list of tuples, each tuple is (x, y)
        """
        tusimple_h_samples = np.arange(160, 720, 10).astype(np.int32)
        tusimple_lanes = []

        for lane in lanes:
            xs = np.ones(720).astype(np.int32) * -2
            for x, y in lane:
                xs[y] = x
            tusimple_xs = xs[tusimple_h_samples].tolist()
            tusimple_lanes.append(tusimple_xs)

        return tusimple_lanes

    def _resize_image_to_tusimple_shape(self, org_img_path, resized_img_path):
        org_img = cv2.imread(org_img_path)
        resized_img = cv2.resize(org_img, (1280, 720))
        cv2.imwrite(resized_img_path, resized_img)

    def convert_annotations_to_tusimple_format(self, output_dir, copy_images=False, sample_interval=10):
        if self.split == 'test':
            raise ValueError("Test set does not have annotations")
        
        h_samples = np.arange(160, 720, 10).astype(np.int32).tolist()

        self.max_lanes = 0
        print("Searching annotation files...")
        json_paths = self.get_json_paths_per_dir()
        print("Found {} subdirectories".format(len(json_paths)))
        processed_cnt = 0
        for subdir_json_paths in json_paths:
            subdir_name = subdir_json_paths['dirname']
            print("Processing subdirectory: {}".format(subdir_name))
            json_paths = subdir_json_paths['json_paths']
            tusimple_label_filename = f'label_llamas_{subdir_name}.json'
            tusimple_label_path = os.path.join(output_dir, tusimple_label_filename)
            tusimple_label_file = open(tusimple_label_path, 'w')

            sample_cnt = 0
            for json_path in tqdm(json_paths):
                if sample_cnt % sample_interval != 0:
                    sample_cnt += 1
                    continue
                sample_cnt += 1
                processed_cnt += 1
                # For one image
                lanes = get_horizontal_values_for_four_lanes(json_path, 
                                                             img_shape=(self.img_h, self.img_w), 
                                                            resized_img_shape=(720, 1280))
                lanes = [[(x, y) for x, y in zip(lane, range(self.img_h)) if x >= 0] for lane in lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]            
                relative_path = self.get_img_path(json_path)
                img_path = os.path.join(self.root, relative_path)
                output_img_path = img_path.split('/')[4:-1]
                output_img_name = img_path.split('/')[-1].replace('_color_rect.png', '_color_rect_tusimple.jpg')
                output_relative_path = os.path.join("clips", *output_img_path, output_img_name)
                if copy_images:
                    # img_path: datasets/llamas/color_images/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/1419280841_0205251000_color_rect.png
                    output_img_file = output_dir + "/" + output_relative_path
                    os.makedirs(os.path.dirname(output_img_file), exist_ok=True)
                    self._resize_image_to_tusimple_shape(img_path, output_img_file)
                #print(lanes)
                tusimple_lanes = self.convert_lanes_to_tusimple_format(lanes)
                tusimple_label = {
                    "raw_file": output_relative_path,
                    "lanes": tusimple_lanes,
                    "h_samples": h_samples,
                }
                tusimple_label_file.write(json.dumps(tusimple_label) + '\n')

        print("Processed {} images".format(processed_cnt))

    # Public method
    def load_annotations(self):
        # the labels are not public for the test set yet
        if self.split == 'test':
            imgs_dir = os.path.join(self.root, TEST_IMGS_DIR)
            self.annotations = [{
                'path': os.path.join(root, file),
                'lanes': [],
                'relative_path': file
            } for root, _, files in os.walk(imgs_dir) for file in files if file.endswith('.png')]
            self.annotations = sorted(self.annotations, key=lambda x: x['path'])
            return
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('cache', exist_ok=True)
        cache_path = 'cache/llamas_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.annotations = pkl.load(cache_file)
                self.max_lanes = max(len(anno['lanes']) for anno in self.annotations)
                return

        self.max_lanes = 0
        print("Searching annotation files...")
        json_paths = self.get_json_paths()
        print('{} annotations found.'.format(len(json_paths)))

        for json_path in tqdm(json_paths):
            lanes = get_horizontal_values_for_four_lanes(json_path) # 返回4条车道线的像素坐标点（经过过滤）
            lanes = [[(x, y) for x, y in zip(lane, range(self.img_h)) if x >= 0] for lane in lanes] # 只保留车道线坐标点
            lanes = [lane for lane in lanes if len(lane) > 0] # remove empty lanes
            relative_path = self.get_img_path(json_path)
            img_path = os.path.join(self.root, relative_path)
            self.max_lanes = max(self.max_lanes, len(lanes))
            self.annotations.append({'path': img_path, 'lanes': lanes, 'aug': False, 'relative_path': relative_path})

        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.annotations, cache_file)

    def assign_class_to_lanes(self, lanes):
        return {label: value for label, value in zip(['l0', 'l1', 'r0', 'r1'], lanes)}

    def get_prediction_string(self, pred):
        """
        一行一条车道线，每个像素行都采样，保留x有效的点；
        """
        ys = np.arange(self.img_h) / self.img_h # 归一化y采样点
        out = []
        for lane in pred:
            xs = lane(ys) # 采样点归一化的x坐标
            valid_mask = (xs >= 0) & (xs < 1) # 过滤无效的坐标点
            xs = xs * self.img_w # 反归一化, [0, 1] -> [0, 1276]
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.img_h # 反归一化, [0, 1] -> [0, 717]
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1] # 反转，从靠近图像底部到顶部
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    # Public method
    def eval_predictions(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            relative_path = self.annotations[idx]['old_anno']['relative_path']
            output_filename = '/'.join(relative_path.split('/')[-2:]).replace('_color_rect.png', '.lines.txt')
            output_filepath = os.path.join(output_basedir, output_filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(output_filepath, 'w') as out_file:
                out_file.write(output)
        if self.split == 'test':
            return {}
        return llamas_metric.eval_predictions(output_basedir, self.labels_dir, unofficial=False)

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
