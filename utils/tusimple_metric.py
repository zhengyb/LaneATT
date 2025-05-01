# pylint: disable-all
import numpy as np
import ujson as json
from sklearn.linear_model import LinearRegression
from functools import partial
import cv2

from p_tqdm import t_map, p_map
from scipy.optimize import linear_sum_assignment
from utils.llamas_metric import discrete_cross_iou, continuous_cross_iou, interpolate_lane

TUSIMPLE_IMG_RES = (720, 1280)


def _culane_metric(pred, anno, raw_file, width=30, iou_threshold=0.4, unofficial=False, img_shape=TUSIMPLE_IMG_RES,
                   draw_img=False, img_path=None):
    """Computes CULane's metric for a single image"""

    pred_ious = np.zeros(len(pred))
    if len(pred) == 0:
        return 0, 0, len(anno), pred_ious, pred_ious > iou_threshold
    if len(anno) == 0:
        return 0, len(pred), 0, pred_ious, pred_ious > iou_threshold
      
    #interp_pred = np.array([interpolate_lane(pred_lane, n=50) for pred_lane in pred])  # (4, 50, 2)
    interp_pred = []
    for pred_lane in pred:
        if len(pred_lane) > 1:
            interp_pred.append(interpolate_lane(pred_lane, n=50))
        else:
            interp_pred.append(np.zeros((50, 2)))    
    interp_pred = np.array(interp_pred)
    #pred = np.array([np.array(pred_lane) for pred_lane in pred], dtype=object)
    anno = np.array([np.array(anno_lane) for anno_lane in anno], dtype=object)

    if unofficial:
        ious = continuous_cross_iou(interp_pred, anno, width=width, img_shape=img_shape)
    else:
        ious = discrete_cross_iou(interp_pred, anno, width=width, img_shape=img_shape)
    #print("ious:")
    #print(ious)
    row_ind, col_ind = linear_sum_assignment(1 - ious)
    #debug
    #print("mached ious:")
    #print(ious[row_ind, col_ind])
    tp = int((ious[row_ind, col_ind] > iou_threshold).sum())
    fp = len(pred) - tp # 误判的车道线
    fn = len(anno) - tp # 漏判的车道线
    pred_ious[row_ind] = ious[row_ind, col_ind]
    #print("pred_ious:")
    #print(pred_ious)
    #print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    return tp, fp, fn, pred_ious, pred_ious > iou_threshold    


class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20 # 直线之间的距离，采样点的像素匹配阈值
    pt_thresh = 0.85 # 车道线匹配率的阈值

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0] # 提取车道线采样点
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs) # 拟合直线
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        # x轴偏差小于thresh，则认为预测正确
        # -100 表示无效值，远大于thresh
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def distances(pred, gt):
        return np.abs(pred - gt)



    @staticmethod
    def bench_f1(pred, gt, y_samples, running_time, get_matches=False):
        """
        """

        pred_lanes = [ list(zip(lane_x, y_samples)) for lane_x in pred]
        gt_lanes = [ list(zip(lane_x, y_samples)) for lane_x in gt]
        new_lanes = []
        for lane in pred_lanes:
            new_lane = [point for point in lane if point[0] >= 0]
            #if len(new_lane) > 0:
            new_lanes.append(new_lane)
        pred_lanes = new_lanes
        new_lanes = []
        for lane in gt_lanes:
            new_lane = [point for point in lane if point[0] >= 0]
            #if len(new_lane) > 0:
            new_lanes.append(new_lane)
        gt_lanes = new_lanes
        #print(f"pred_lanes: {pred_lanes}")
        #print(f"gt_lanes: {gt_lanes}")
        tp, fp, fn, ious, matches = _culane_metric(pred_lanes, gt_lanes, None, unofficial=False)
        return tp, fp, fn, matches, ious, None

    @staticmethod
    def bench(pred, gt, y_samples, running_time, get_matches=False):
        """
        input:
            pred: 预测的车道线
            gt: 真实的车道线
            y_samples: 车道线采样点
            running_time: 运行时间
            get_matches: 是否获取匹配结果
        output:
            accuracy: 准确率
            fp: 误判的车道线
            fn: 漏判的车道线
            my_matches: 匹配结果
            my_accs: 匹配准确率
            my_dists: 匹配距离
        """
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception('Format of lanes error.')
        if running_time > 20000 or len(gt) + 2 < len(pred):
            if get_matches:
                return 0., 0., 1., [False] * len(pred), [0] * len(pred), [None] * len(pred)
            return 0., 0., 1.,
        # 采用直线拟合计算车道线的角度
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        # 计算车道线预测的阈值
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.
        my_matches = [False] * len(pred)
        my_accs = [0] * len(pred)
        my_dists = [None] * len(pred)
        for x_gts, thresh in zip(gt, threshs):
            # 计算当前真实车道线x_gts与所有预测车道线x_preds的匹配准确率
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            my_accs = np.maximum(my_accs, accs) # 逐元素比较两个数组，保留每个位置的最大值？
            max_acc = np.max(accs) if len(accs) > 0 else 0. # 预测车道线中准确率最高的
            # 计算当前真实车道线x_gts与所有预测车道线x_preds的距离
            my_dist = [LaneEval.distances(np.array(x_preds), np.array(x_gts)) for x_preds in pred]
            if len(accs) > 0:
                # np.argmax(accs): 预测车道线中匹配准确率最高的索引
                my_dists[np.argmax(accs)] = {
                    'y_gts': list(np.array(y_samples)[np.array(x_gts) >= 0].astype(int)), # 仅保留有车道线的y sample
                    'dists': list(my_dist[np.argmax(accs)]) # 当前真实车道线x_gts与预测车道线x_preds的距离
                }

            if max_acc < LaneEval.pt_thresh: # 85%
                fn += 1 
            else:
                my_matches[np.argmax(accs)] = True
                matched += 1 # True Positive
            line_accs.append(max_acc)
        fp = len(pred) - matched # False Positive,误判的车道线
        if len(gt) > 4 and fn > 0:
            fn -= 1 # 修正False Negative， why?
        s = sum(line_accs)
        if len(gt) > 4: # 只考虑匹配准确率最高的4条车道线, laneATT只考虑4条车道线
            s -= min(line_accs)
        if get_matches:
            return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(
                min(len(gt), 4.), 1.), my_matches, my_accs, my_dists
        return s / max(min(4.0, len(gt)), 1.), fp / len(pred) if len(pred) > 0 else 0., fn / max(min(len(gt), 4.), 1.)


    @staticmethod
    def bench_one_submit_f1(pred_file, gt_file, save_img=True):
        """
        bench_one_submit_f1 函数用于计算预测车道线与真实车道线的F1得分。
        input:
            pred_file: 预测的车道线
            gt_file: 真实的车道线
        output:
            f1: f1得分
        """
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {img['raw_file']: img for img in json_gt}
        total_tp, total_fp, total_fn = 0., 0., 0.
        run_times = []
        predictions = []
        annotations = []
        raw_files = []
        for pred in json_pred:
            # for each image
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']

            # format gt_lanes & pred_lanes
            iou_gt_lanes = []
            for lane in gt_lanes:
                lane_ious = [(x, y) for x, y in zip(lane, y_samples) if x >= 0]
                if len(lane_ious) < 2:
                    continue
                iou_gt_lanes.append(lane_ious)
            iou_pred_lanes = []
            for lane in pred_lanes:
                lane_ious = [(x, y) for x, y in zip(lane, y_samples) if x >= 0]
                if len(lane_ious) < 2:
                    continue
                iou_pred_lanes.append(lane_ious)
            predictions.append(iou_pred_lanes)
            annotations.append(iou_gt_lanes)
            if save_img:
                raw_files.append(raw_file)
            else:
                raw_files.append(None)
        #print(f"raw_files: {raw_files[-1]}")
        #print(f"predictions: {predictions[-1]}")
        #print(f"annotations: {annotations[-1]}")
        results = p_map(partial(_culane_metric, width=30, unofficial=False, img_shape=TUSIMPLE_IMG_RES),
                        predictions, annotations, raw_files)
        num = len(gts)
        total_tp = sum(tp for tp, _, _, _, _ in results)
        total_fp = sum(fp for _, fp, _, _, _ in results)
        total_fn = sum(fn for _, _, fn, _, _ in results)
        if total_tp == 0:
            precision = 0
            recall = 0
            f1 = 0
        else:
            precision = float(total_tp) / (total_tp + total_fp)
            recall = float(total_tp) / (total_tp + total_fn)
            f1 = 2 * precision * recall / (precision + recall)      
        return json.dumps([{
            'name': 'F1',
            'value': f1,
            'order': 'desc'
        }, {
            'name': 'Precision',
            'value': precision,
            'order': 'desc'
        }, {
            'name': 'Recall',
            'value': recall,
            'order': 'desc'
        }, {
            'name': 'FPS',
            'value': 1000. / np.mean(run_times),
            'order': 'desc'
        }, {
            'name': 'TP',
            'value': total_tp,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': total_fp,
            'order': 'desc'
        }, {
            'name': 'FN',
            'value': total_fn,
            'order': 'desc'
        }])
    
    @staticmethod
    def bench_one_submit(pred_file, gt_file):
        try:
            json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        except BaseException as e:
            raise Exception('Fail to load json file of the prediction.')
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise Exception('We do not get the predictions of all the test tasks')
        gts = {img['raw_file']: img for img in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        run_times = []
        for pred in json_pred:
            if 'raw_file' not in pred or 'lanes' not in pred or 'run_time' not in pred:
                raise Exception('raw_file or lanes or run_time not in some predictions.')
            raw_file = pred['raw_file']
            pred_lanes = pred['lanes']
            run_time = pred['run_time']
            run_times.append(run_time)
            if raw_file not in gts:
                raise Exception('Some raw_file from your predictions do not exist in the test tasks.')
            gt = gts[raw_file]
            gt_lanes = gt['lanes']
            y_samples = gt['h_samples']
            try:
                a, p, n = LaneEval.bench(pred_lanes, gt_lanes, y_samples, run_time)
            except BaseException as e:
                raise Exception('Format of lanes error.')
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        # the first return parameter is the default ranking parameter
        return json.dumps([{
            'name': 'Accuracy',
            'value': accuracy / num,
            'order': 'desc'
        }, {
            'name': 'FP',
            'value': fp / num,
            'order': 'asc'
        }, {
            'name': 'FN',
            'value': fn / num,
            'order': 'asc'
        }, {
            'name': 'FPS',
            'value': 1000. / np.mean(run_times)
        }])


if __name__ == '__main__':
    import sys

    try:
        if len(sys.argv) != 3:
            raise Exception('Invalid input arguments')
        print(LaneEval.bench_one_submit(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(e)
        # sys.exit(e.message)
