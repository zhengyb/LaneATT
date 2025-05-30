import logging

import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmenters import Resize
from torchvision.transforms import ToTensor
from torch.utils.data.dataset import Dataset
from scipy.interpolate import InterpolatedUnivariateSpline
from imgaug.augmentables.lines import LineString, LineStringsOnImage

from lib.lane import Lane

from .culane import CULane
from .tusimple import TuSimple
from .llamas import LLAMAS
from .nolabel_dataset import NoLabelDataset

GT_COLOR = (255, 0, 0) # blue
PRED_HIT_COLOR = (0, 255, 0) # green
PRED_MISS_COLOR = (0, 0, 255) # red
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) # 图像归一化均值
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) # 图像归一化标准差


class LaneDataset(Dataset):
    def __init__(self,
                 S=72,
                 dataset='tusimple',
                 augmentations=None,
                 normalize=False,
                 img_size=(360, 640),
                 aug_chance=1.,
                 **kwargs):
        super(LaneDataset, self).__init__()
        if dataset == 'tusimple':
            self.dataset = TuSimple(**kwargs)
        elif dataset == 'culane':
            self.dataset = CULane(**kwargs)
        elif dataset == 'llamas':
            self.dataset = LLAMAS(**kwargs)
        elif dataset == 'nolabel_dataset':
            self.dataset = NoLabelDataset(**kwargs)
        else:
            raise NotImplementedError()
        self.name = self.dataset.name
        self.n_strips = S - 1 # 71条y轴分割线，不含0和360
        self.n_offsets = S # 72个y轴偏移量
        self.normalize = normalize
        self.img_h, self.img_w = img_size
        self.strip_size = self.img_h / self.n_strips # y轴分割带大小： 360/71 = 5
        self.logger = logging.getLogger(__name__)

        # y at each x offset, 73个ys
        self.offsets_ys = np.arange(self.img_h, -1, -self.strip_size) # arange(360, -1, -5): [360, 355, 350 ... 0]
        self.transform_annotations()

        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        else:
            augmentations = []

        transformations = iaa.Sequential([Resize({'height': self.img_h, 'width': self.img_w})])
        self.to_tensor = ToTensor()
        self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        self.max_lanes = self.dataset.max_lanes

    @property
    def annotations(self):
        return self.dataset.annotations

    def transform_annotations(self):
        self.logger.info("Transforming annotations to the model's target format...")
        self.dataset.annotations = np.array(list(map(self.transform_annotation, self.dataset.annotations)))
        self.logger.info('Done.')

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    def transform_annotation(self, anno, img_wh=None):
        if img_wh is None:
            img_h = self.dataset.get_img_heigth(anno['path'])
            img_w = self.dataset.get_img_width(anno['path'])
        else:
            img_w, img_h = img_wh

        old_lanes = anno['lanes']

        # removing lanes with less than 2 points
        old_lanes = filter(lambda x: len(x) > 1, old_lanes)
        # sort lane points by Y (bottom to top of the image)
        old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
        # remove points with same Y (keep first occurrence)
        old_lanes = [self.filter_lane(lane) for lane in old_lanes]
        # normalize the annotation coordinates. original sizes -> (360, 640)
        old_lanes = [[[x * self.img_w / float(img_w), y * self.img_h / float(img_h)] for x, y in lane]
                     for lane in old_lanes]
        # create tranformed annotations
        lanes = np.ones((self.dataset.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
                        dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S coordinates
        # lanes are invalid by default
        lanes[:, 0] = 1
        lanes[:, 1] = 0
        for lane_idx, lane in enumerate(old_lanes):
            try:
                xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
            except AssertionError:
                continue
            if len(xs_inside_image) == 0:
                continue
            all_xs = np.hstack((xs_outside_image, xs_inside_image))
            lanes[lane_idx, 0] = 0
            lanes[lane_idx, 1] = 1
            lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips # 1 - start_y: 归一化到[0,1]
            lanes[lane_idx, 3] = xs_inside_image[0] # start_x: 未归一化
            lanes[lane_idx, 4] = len(xs_inside_image) # length
            # 车道线采样点x坐标，隐含y坐标：对应的y坐标从360开始，以5为步进递减； 通常不包含ys=0的点
            lanes[lane_idx, 5:5 + len(all_xs)] = all_xs 
        new_anno = {'path': anno['path'], 'label': lanes, 'old_anno': anno}
        return new_anno

    def sample_lane(self, points, sample_ys):
        # this function expects the points to be sorted
        # sample_ys: [360, 355, 350, ... 0]
        points = np.array(points)
        if not np.all(points[1:, 1] < points[:-1, 1]):
            raise Exception('Annotaion points have to be sorted')
        x, y = points[:, 0], points[:, 1]

        # interpolate points inside domain
        # 2. 3次样条插值处理器（处理车道线中间部分）
        assert len(points) > 1
        # y[::-1], x[::-1]  反转坐标序列
        interp = InterpolatedUnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
        # 反转坐标使y递增，满足样条插值要求

        # 3. 确定插值域范围ys
        domain_min_y = y.min()
        domain_max_y = y.max()
        sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
        assert len(sample_ys_inside_domain) > 0
        interp_xs = interp(sample_ys_inside_domain) # 执行插值

        # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
        # 4. 线性外推（处理图像底部区域）
        two_closest_points = points[:2]
        extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=1)
        extrap_ys = sample_ys[sample_ys > domain_max_y]
        extrap_xs = np.polyval(extrap, extrap_ys)
        # 5. 合并结果
        all_xs = np.hstack((extrap_xs, interp_xs)) # 外推结果在前，插值结果在后
        # [360, 355, ..., 325（外推）, 320（插值）, 315, ..., 0]

        # separate between inside and outside points
        # 6. 划分有效/无效点
        inside_mask = (all_xs >= 0) & (all_xs < self.img_w) # 判断x坐标是否在图像范围内
        xs_inside_image = all_xs[inside_mask]
        xs_outside_image = all_xs[~inside_mask]

        return xs_outside_image, xs_inside_image

    def label_to_lanes(self, label):
        lanes = []
        for l in label:
            if l[1] == 0:
                continue
            xs = l[5:] / self.img_w
            ys = self.offsets_ys / self.img_h
            start = int(round(l[2] * self.n_strips))
            length = int(round(l[4]))
            xs = xs[start:start + length][::-1]
            ys = ys[start:start + length][::-1]
            xs = xs.reshape(-1, 1)
            ys = ys.reshape(-1, 1)
            points = np.hstack((xs, ys))

            lanes.append(Lane(points=points))
        return lanes

    def draw_annotation(self, idx, label=None, pred=None, img=None):
        # Get image if not provided
        if img is None:
            # print(self.annotations[idx]['path'])
            img, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
            img = img.permute(1, 2, 0).numpy() # HWC
            if self.normalize:
                img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
            img = (img * 255).astype(np.uint8)
        else:
            _, label, _ = self.__getitem__(idx)
            label = self.label_to_lanes(label)
        img = cv2.resize(img, (self.img_w, self.img_h))

        img_h, _, _ = img.shape
        # Pad image to visualize extrapolated predictions
        pad = 0
        if pad > 0:
            img_pad = np.zeros((self.img_h + 2 * pad, self.img_w + 2 * pad, 3), dtype=np.uint8)
            img_pad[pad:-pad, pad:-pad, :] = img
            img = img_pad
        data = [(None, None, label)]
        if pred is not None:
            # print(len(pred), 'preds')
            fp, fn, matches, accs = self.dataset.get_metrics(pred, idx)
            # print('fp: {} | fn: {}'.format(fp, fn))
            # print(len(matches), 'matches')
            # print(matches, accs)
            assert len(matches) == len(pred), f"matches length: {len(matches)}, pred length: {len(pred)}"
            data.append((matches, accs, pred))
        else:
            fp = fn = None
        # 示例：
        # data = [
        #     (None, None, label),
        #     (matches, accs, pred)
        # ]
        for matches, accs, datum in data:
            for i, l in enumerate(datum):
                if matches is None:
                    color = GT_COLOR
                    acc = ''
                elif matches[i]: # pred
                    color = PRED_HIT_COLOR
                    acc = str(round(accs[i], 2))
                else: # pred miss
                    color = PRED_MISS_COLOR
                    acc = str(round(accs[i], 2))
                points = l.points # 归一化坐标
                points[:, 0] *= img.shape[1] # 转换为像素坐标
                points[:, 1] *= img.shape[0]
                points = points.round().astype(int) # 取整
                points += pad # 添加填充偏移
                xs, ys = points[:, 0], points[:, 1]
                # 用相邻点连线绘制2D车道线
                if len(points) > 0:
                    text_point = points[0]
                    img = cv2.putText(img, acc, (text_point[0] + pad, text_point[1] + pad),
                                  fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                  fontScale=0.7,
                                  color=GT_COLOR)
                
                for curr_p, next_p in zip(points[:-1], points[1:]):
                    img = cv2.line(img,
                                   tuple(curr_p),
                                   tuple(next_p),
                                   color=color,
                                   #thickness=30)
                                   thickness=3 if matches is None else 3)
                # if 'start_x' in l.metadata:
                #     start_x = l.metadata['start_x'] * img.shape[1]
                #     start_y = l.metadata['start_y'] * img.shape[0]
                #     cv2.circle(img, (int(start_x + pad), int(img_h - 1 - start_y + pad)),
                #                radius=5,
                #                color=(0, 0, 255),
                #                thickness=-1)
                # if len(xs) == 0:
                #     print("Empty pred")
                # if len(xs) > 0 and accs is not None:
                #     cv2.putText(img,
                #                 '{:.0f} ({})'.format(accs[i] * 100, i),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=color)
                #     cv2.putText(img,
                #                 '{:.0f}'.format(l.metadata['conf'] * 100),
                #                 (int(xs[len(xs) // 2] + pad), int(ys[len(xs) // 2] + pad - 50)),
                #                 fontFace=cv2.FONT_HERSHEY_COMPLEX,
                #                 fontScale=0.7,
                #                 color=(255, 0, 255))
        return img, fp, fn

    def lane_to_linestrings(self, lanes):
        lines = []
        for lane in lanes:
            lines.append(LineString(lane))

        return lines

    def linestrings_to_lanes(self, lines):
        lanes = []
        for line in lines:
            lanes.append(line.coords)

        return lanes

    def __getitem__(self, idx):
        # 1. 读取原始图像和标注
        item = self.dataset[idx]
        img_org = cv2.imread(item['path']) # BGR
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB) # 转换为RGB
        line_strings_org = self.lane_to_linestrings(item['old_anno']['lanes']) # 读取数据集原始标签
        line_strings_org = LineStringsOnImage(line_strings_org, shape=img_org.shape) # 将标注转换为LineStringsOnImage对象s

        # 2. 数据增强
        for i in range(30):
            # 3. 数据增强变换
            img, line_strings = self.transform(image=img_org.copy(), line_strings=line_strings_org)
            line_strings.clip_out_of_image_() # 裁剪超出图像边界的标注

            # 4. 将标注转换为Lane对象
            new_anno = {'path': item['path'], 'lanes': self.linestrings_to_lanes(line_strings)}
            try:
                # 5. 将标注转换为模型目标格式
                label = self.transform_annotation(new_anno, img_wh=(self.img_w, self.img_h))['label']
                break
            except:
                if (i + 1) == 30:
                    self.logger.critical('Transform annotation failed 30 times :(')
                    exit()

        # 6. 图像归一化
        img = img / 255. # 归一化到[0, 1]
        if self.normalize: # LLAMAS和TuSimple数据集未使用归一化
            img = (img - IMAGENET_MEAN) / IMAGENET_STD # ImageNet标准化
        img = self.to_tensor(img.astype(np.float32)) # 转换为Tensor
        return (img, label, idx)

    def __len__(self):
        return len(self.dataset)
