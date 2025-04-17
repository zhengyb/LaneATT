# pylint: disable-all
import numpy as np
import ujson as json
from sklearn.linear_model import LinearRegression


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
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def distances(pred, gt):
        return np.abs(pred - gt)

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
