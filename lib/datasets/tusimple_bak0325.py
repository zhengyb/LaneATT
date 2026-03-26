import os
import json
import random
import logging
import pickle as pkl
import numpy as np

from utils.tusimple_metric import LaneEval

from .lane_dataset_loader import LaneDatasetLoader

"""

 """
SPLIT_FILES = {
    "train+val": [
        "label_data_0313.json",
        "label_data_0601.json",
        "label_data_0531.json",
        # convert from LLAMAS
        "label_llamas_images-2014-12-18-14-17-05.json",
        "label_llamas_images-2014-12-18-14-28-45.json",
        "label_llamas_images-2014-12-18-14-43-48.json",
        "label_llamas_images-2014-12-22-13-04-51_mapping_280N_2nd_lane.json",
        "label_llamas_images-2014-12-22-14-01-36_mapping_280N_3rd_lane.json",
        "label_llamas_images-2014-12-22-12-02-45_mapping_280N_ramps.json",
        "label_llamas_images-2014-12-22-14-36-42_mapping_280N_4th_lane.json",
        "label_llamas_images-2014-12-22-12-35-10_mapping_280S_ramps.json",
        "label_llamas_images-2014-12-22-15-18-11_mapping_RTC_to_280_left_lanes.json",
    ],
    "train": [
        "label_data_0313_20250501.json",
        "label_data_0601_20250501.json",
        # convert from LLAMAS
        "label_llamas_images-2014-12-18-14-17-05.json",
        "label_llamas_images-2014-12-22-13-04-51_mapping_280N_2nd_lane.json",
        "label_llamas_images-2014-12-18-14-43-48.json",
        "label_llamas_images-2014-12-22-12-02-45_mapping_280N_ramps.json",
        "label_llamas_images-2014-12-22-14-36-42_mapping_280N_4th_lane.json",
        "label_llamas_images-2014-12-22-15-18-11_mapping_RTC_to_280_left_lanes.json",
    ],
    "val": [
        "label_data_0531_20250501.json",
        # convert from LLAMAS
        "label_llamas_images-2014-12-18-14-28-45.json",
        "label_llamas_images-2014-12-22-14-01-36_mapping_280N_3rd_lane.json",
        "label_llamas_images-2014-12-22-12-35-10_mapping_280S_ramps.json",
    ],
    "test": [
        #"test_label_20250501.json",
        # convert from LLAMAS
        "label_llamas_images-2014-12-22-12-35-10_mapping_280S_ramps.json",
        "label_llamas_images-2014-12-22-14-19-07_mapping_280S_3rd_lane.json",
    ],
    "test_curve_case": [
    ],
    "test_extreme_weather_case": [
    ],
    "test_night_case": [
    ],
    "test_intersection_case": [
    ],
    "test_up_down_case": [
    ],
    "test_merge_split_case": [
    ],
    "test_highway_case": [
        #"test_label_20250501.json",
        # convert from LLAMAS
        "label_llamas_images-2014-12-22-12-35-10_mapping_280S_ramps.json",
        "label_llamas_images-2014-12-22-14-19-07_mapping_280S_3rd_lane.json",
    ],    
    
    
}


class TuSimple(LaneDatasetLoader):
    def __init__(self, split="train", max_lanes=None, root=None):
        self.name = 'tusimple'
        self.split = split
        self.root = root
        self.logger = logging.getLogger(__name__)

        if split not in SPLIT_FILES.keys():
            raise Exception("Split `{}` does not exist.".format(split))

        self.anno_files = [os.path.join(self.root, path) for path in SPLIT_FILES[split]]
        if split in ['test', 'train', 'val']:
            self.add_openlane_anno_files(self.root, split)
        elif split in ['test_highway_case']:
            pass
        elif split in ['test_curve_case', 'test_extreme_weather_case', 
                       'test_night_case', 'test_intersection_case', 
                       'test_up_down_case', 'test_merge_split_case']:
            self.get_scene_anno_files(self.root, split)

        if root is None:
            raise Exception("Please specify the root directory")

        self.img_w, self.img_h = 1280, 720
        self.annotations = []
        self.load_annotations()

        # Force max_lanes, used when evaluating testing with models trained on other datasets
        if max_lanes is not None:
            self.max_lanes = max_lanes

    def get_scene_anno_files(self, root_dir, split):
        scene_anno_filename = split.replace('test_', 'ol_scene_') + '.json'
        self.anno_files = [os.path.join(root_dir, scene_anno_filename)]

    def add_openlane_anno_files(self, root_dir, split):
        openlane_anno_files = []
        if split == "train":
            anno_prefix = "ol_training"
        elif split == "val":
            anno_prefix = "ol_validation"
        elif split == "test":
            anno_prefix = "ol_test"
        else:
            print(f"Split `{split}` does not exist.")
            return
            #    raise Exception("Split `{}` does not exist.".format(split))
        for anno_file in os.listdir(root_dir):
            if anno_file.endswith(".json") and anno_file.startswith(anno_prefix):
                openlane_anno_files.append(os.path.join(root_dir, anno_file))
        self.anno_files.extend(openlane_anno_files)

    def get_img_heigth(self, _):
        return 720

    def get_img_width(self, _):
        return 1280

    def get_metrics(self, lanes, idx, use_f1=True):
        label = self.annotations[idx]
        org_anno = label["old_anno"]
        pred = self.pred2lanes(org_anno["path"], lanes, org_anno["y_samples"])
        if use_f1:
            #print(f"path: {org_anno['path']}")
            #print(f"org_anno['lanes']: {org_anno['org_lanes']}")
            _, fp, fn, matches, ious_accs, _ = LaneEval.bench_f1(
                pred, org_anno["org_lanes"], org_anno["y_samples"], 0, True
            )
        else:
            _, fp, fn, matches, ious_accs, _ = LaneEval.bench(
                pred, org_anno["org_lanes"], org_anno["y_samples"], 0, True
            )
        return fp, fn, matches, ious_accs

    def pred2lanes(self, path, pred, y_samples):
        # pred to tusimple annotation for 1 image
        ys = np.array(y_samples) / self.img_h  # 归一化到[0,1]
        lanes = []
        for lane in pred:
            xs = lane(ys)  # 采样点归一化的x坐标
            # xs > 1 is added by Reuben.
            invalid_mask = (xs < 0) | (xs > 1)  # 背景点mask
            #invalid_mask = (xs < 0)
            lane = (xs * self.get_img_width(path)).astype(int)  # 反归一化到图像像素坐标
            lane[invalid_mask] = -2  # 背景点赋值为-2
            #invalid_mask = (lane < 0)
            #invalid_count = np.sum(invalid_mask)
            #if len(ys) - invalid_count < 2:
            #    continue
            lanes.append(lane.tolist())

        return lanes

    def load_annotations(self):
        self.logger.info("Loading TuSimple annotations...")
        self.annotations = []
        # Waiting for the dataset to load is tedious, let's cache it
        os.makedirs('anno_cache', exist_ok=True)
        cache_path = 'anno_cache/tusimple_{}.pkl'.format(self.split)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.annotations = pkl.load(cache_file)
                self.max_lanes = max(len(anno['lanes']) for anno in self.annotations)
                print(f"Loaded {len(self.annotations)} annotations from cached file {cache_path}, MAX. lanes {self.max_lanes}")
                return
                    
        max_lanes = 0
        for anno_file in self.anno_files:  # 遍历所有标签文件
            with open(anno_file, "r") as anno_obj:  # 打开标签文件
                lines = anno_obj.readlines()
            for line in lines:  # 遍历每一行，1个image
                data = json.loads(line)
                y_samples = data["h_samples"]  # 1个image的y坐标
                gt_lanes = data["lanes"]  # 1个image的gt_lanes

                # 过滤掉没有gt_lanes的image
                #if len(gt_lanes) == 0:
                #    continue
                # 过滤掉gt_lanes大于5条的image, openlane数据集的标注有问题???
                if len(gt_lanes) > 5:
                    continue

                lanes = [
                    [(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                    for lane in gt_lanes
                ]
                lanes = [
                    lane for lane in lanes if len(lane) > 0
                ]  # 去掉没有采样点的lane
                max_lanes = max(max_lanes, len(lanes))
                self.annotations.append(
                    {
                        "path": os.path.join(self.root, data["raw_file"]),
                        "org_path": data["raw_file"],
                        "org_lanes": gt_lanes,
                        "lanes": lanes,
                        "aug": False,
                        "y_samples": y_samples,
                    }
                )
            print(f"Loaded anno_file: {anno_file}")

        if self.split == "train":
            random.shuffle(self.annotations)
        self.max_lanes = max_lanes
        self.logger.info(
            "%d annotations loaded, with a maximum of %d lanes in an image.",
            len(self.annotations),
            self.max_lanes,
        )
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.annotations, cache_file)        

    def random_sample_annotations(self, sample_num, new_dataset_dir):
        import shutil

        sampled_annotations = random.sample(self.annotations, sample_num)
        sampled_anno_filename = os.path.join(new_dataset_dir, f"sampled_anno_{self.split}.json")
        new_image_dir = os.path.join(new_dataset_dir, "images", self.split)
        if os.path.exists(new_image_dir):
            shutil.rmtree(new_image_dir)
        os.makedirs(new_image_dir, exist_ok=True)
        with open(sampled_anno_filename, "w") as f:
            for idx in range(len(sampled_annotations)):
                anno = sampled_annotations[idx]
                old_image_path = anno["org_path"]
                new_image_file = f"{idx:06d}.jpg"
                new_image_path = os.path.join(new_image_dir, new_image_file)
                # TODO: copy image to new_dataset_dir
                shutil.copy(os.path.join(self.root, old_image_path), new_image_path)
                tusimple_anno = {
                    "raw_file": os.path.join("images", self.split, new_image_file),
                    "h_samples": anno["y_samples"],
                    "lanes": anno["org_lanes"],
                }
                f.write(json.dumps(tusimple_anno))
                f.write("\n")

        print(f"Sampled {sample_num} annotations from {self.split} set of TuSimple dataset")
        return sampled_annotations

    def transform_annotations(self, transform):
        self.annotations = list(map(transform, self.annotations))

    def pred2tusimpleformat(self, idx, pred, runtime):
        # 1 image
        runtime *= 1000.0  # s to ms
        img_name = self.annotations[idx]["old_anno"]["org_path"]
        h_samples = self.annotations[idx]["old_anno"]["y_samples"]
        lanes = self.pred2lanes(img_name, pred, h_samples)
        output = {"raw_file": img_name, "lanes": lanes, "run_time": runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.0e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, "w") as output_file:
            output_file.write("\n".join(lines))

    def eval_predictions(self, predictions, output_basedir, runtimes=None, use_f1=True):
        pred_filename = os.path.join(output_basedir, "tusimple_predictions.json")
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        # merge anno_files
        if len(self.anno_files) > 1:
            merged_anno_file = os.path.join(output_basedir, "merged_anno.json")
            with open(merged_anno_file, "w") as of:
                for anno_file in self.anno_files:
                    with open(anno_file, "r") as f:
                        of.write(f.read())
            merged_anno = merged_anno_file
        else:
            merged_anno = self.anno_files[0]

        if use_f1:
            result = json.loads(LaneEval.bench_one_submit_f1(pred_filename, merged_anno))
        else:
            result = json.loads(LaneEval.bench_one_submit(pred_filename, merged_anno))
            #  {'Accuracy': 0.9565514103730626, 'FP': 0.0803883295664674, 'FN': 0.04018560372577228, 'FPS': 1000.0}
        
        table = {}
        for metric in result:
            table[metric["name"]] = metric["value"]
        return table

    def __getitem__(self, idx):
        return self.annotations[idx]

    def __len__(self):
        return len(self.annotations)
