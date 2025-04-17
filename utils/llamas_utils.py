# All following lines (which were slightly modified) were taken from: https://github.com/karstenBehrendt/unsupervised_llamas
# Its license is copied here

# ##### Begin License ######
# MIT License

# Copyright (c) 2019 Karsten Behrendt, Robert Bosch LLC

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ##### End License ######

# Start code under the previous license

import os
import json
import numpy as np


def get_files_from_folder(directory, extension=None):
    """Get all files within a folder that fit the extension """
    # NOTE Can be replaced by glob for newer python versions
    label_files = []
    for root, _, files in os.walk(directory):
        for some_file in files:
            label_files.append(os.path.abspath(os.path.join(root, some_file)))
    if extension is not None:
        label_files = list(filter(lambda x: x.endswith(extension), label_files))
    return label_files


def get_label_base(label_path):
    """ Gets directory independent label path """
    return '/'.join(label_path.split('/')[-2:])


def get_labels(dataset_root, split='test'):
    """ Gets label files of specified dataset split """
    label_paths = get_files_from_folder(os.path.join(dataset_root, split), '.json')
    return label_paths


def _extend_lane(lane, projection_matrix):
    """Extends marker closest to the camera

    Adds an extra marker that reaches the end of the image. 假设最近的车道线标记是直线.

    Parameters
    ----------
    lane : iterable of markers
    projection_matrix : 3x3 projection matrix
    """
    # Unfortunately, we did not store markers beyond the image plane. That hurts us now
    # z is the orthongal distance to the car. It's good enough

    # The markers are automatically detected, mapped, and labeled. There exist faulty ones,
    # e.g., horizontal markers which need to be filtered

    # 过滤水平标记和垂直标记。（垂直标记为什么要过滤?)
    filtered_markers = filter(
        lambda x: (x['pixel_start']['y'] != x['pixel_end']['y'] and x['pixel_start']['x'] != x['pixel_end']['x']),
        lane['markers'])
    # might be the first marker in the list but not guaranteed
    closest_marker = min(filtered_markers, key=lambda x: x['world_start']['z']) # 最近的车道线标记

    if closest_marker['world_start']['z'] < 0:  # This one likely equals "if False"
        return lane

    # World marker extension approximation
    x_gradient = (closest_marker['world_end']['x'] - closest_marker['world_start']['x']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])
    y_gradient = (closest_marker['world_end']['y'] - closest_marker['world_start']['y']) /\
        (closest_marker['world_end']['z'] - closest_marker['world_start']['z'])

    # 通过线性外推计算z=1时的x, y坐标（假设z坐标向相机方向延伸）
    zero_x = closest_marker['world_start']['x'] - (closest_marker['world_start']['z'] - 1) * x_gradient
    zero_y = closest_marker['world_start']['y'] - (closest_marker['world_start']['z'] - 1) * y_gradient

    # Pixel marker extension approximation
    pixel_x_gradient = (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x']) /\
        (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y'])
    pixel_y_gradient = (closest_marker['pixel_end']['y'] - closest_marker['pixel_start']['y']) /\
        (closest_marker['pixel_end']['x'] - closest_marker['pixel_start']['x'])

    # 在图像坐标系，通过线性外推计算y=716时的x坐标
    pixel_zero_x = closest_marker['pixel_start']['x'] + (716 - closest_marker['pixel_start']['y']) * pixel_x_gradient
    if pixel_zero_x < 0:
        # 计算x=0时的y坐标
        left_y = closest_marker['pixel_start']['y'] - closest_marker['pixel_start']['x'] * pixel_y_gradient
        new_pixel_point = (0, left_y)
    elif pixel_zero_x > 1276:
        # 计算x=1276时的y坐标
        right_y = closest_marker['pixel_start']['y'] + (1276 - closest_marker['pixel_start']['x']) * pixel_y_gradient
        new_pixel_point = (1276, right_y)
    else:
        # 计算y=716时的x坐标
        new_pixel_point = (pixel_zero_x, 716)

    new_marker = {
        'lane_marker_id': 'FAKE',
        'world_end': {
            'x': closest_marker['world_start']['x'],
            'y': closest_marker['world_start']['y'],
            'z': closest_marker['world_start']['z']
        },
        'world_start': {
            'x': zero_x,
            'y': zero_y,
            'z': 1
        },
        'pixel_end': {
            'x': closest_marker['pixel_start']['x'],
            'y': closest_marker['pixel_start']['y']
        },
        'pixel_start': {
            'x': ir(new_pixel_point[0]),
            'y': ir(new_pixel_point[1])
        }
    }
    lane['markers'].insert(0, new_marker)

    return lane


class SplineCreator():
    """
    For each lane divder
      - all lines are projected
      - linearly interpolated to limit oscillations
      - interpolated by a spline
      - subsampled to receive individual pixel values

    The spline creation can be optimized!
      - Better spline parameters
      - Extend lowest marker to reach bottom of image would also help
      - Extending last marker may in some cases be interesting too
    Any help is welcome.

    Call create_all_points and get the points in self.sampled_points
    It has an x coordinate for each value for each lane

    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.json_content = read_json(json_path)
        self.lanes = self.json_content['lanes']
        self.lane_marker_points = {}
        self.sampled_points = {}  # <--- the interesting part
        self.debug_image = np.zeros((717, 1276, 3), dtype=np.uint8)

    def _sample_points(self, lane, ypp=5, between_markers=True):
        """ Markers are given by start and endpoint. This one adds extra points
        which need to be considered for the interpolation. Otherwise the spline
        could arbitrarily oscillate between start and end of the individual markers

        Parameters
        ----------
        lane: polyline, in theory but there are artifacts which lead to inconsistencies
              in ordering. There may be parallel lines. The lines may be dashed. It's messy.
        ypp: y-pixels per point, e.g. 10 leads to a point every ten pixels
        between_markers : bool, interpolates inbetween dashes

        Notes
        -----
        Especially, adding points in the lower parts of the image (high y-values) because
        the start and end points are too sparse.
        Removing upper lane markers that have starting and end points mapped into the same pixel.
        """

        # Collect all x values from all markers along a given line. There may be multiple
        # intersecting markers, i.e., multiple entries for some y values
        # Step1: 在车道线内部，按直线作插值.
        x_values = [[] for i in range(717)]
        for marker in lane['markers']:
            x_values[marker['pixel_start']['y']].append(marker['pixel_start']['x'])

            height = marker['pixel_start']['y'] - marker['pixel_end']['y']
            if height > 2:
                slope = (marker['pixel_end']['x'] - marker['pixel_start']['x']) / height
                step_size = (marker['pixel_start']['y'] - marker['pixel_end']['y']) / float(height)
                for i in range(height + 1):
                    x = marker['pixel_start']['x'] + slope * step_size * i
                    y = marker['pixel_start']['y'] - step_size * i
                    x_values[ir(y)].append(ir(x))

        # Calculate average x values for each y value
        for y, xs in enumerate(x_values):
            if not xs:
                x_values[y] = -1
            else:
                x_values[y] = sum(xs) / float(len(xs))

        # In the following, we will only interpolate between markers if needed
        if not between_markers:
            return x_values  # TODO ypp

        # Step2: 用直线插值填补marker之间缺失的点
        # # interpolate between markers
        current_y = 0
        while x_values[current_y] == -1:  # skip missing first entries
            current_y += 1

        # Also possible using numpy.interp when accounting for beginning and end
        next_set_y = 0
        try:
            while current_y < 717:
                if x_values[current_y] != -1:  # set. Nothing to be done
                    current_y += 1
                    continue

                # Finds target x value for interpolation
                while next_set_y <= current_y or x_values[next_set_y] == -1:
                    next_set_y += 1
                    if next_set_y >= 717:
                        raise StopIteration
                # 此时，next_set_y 是下一个marker的起始y坐标
                # 直线插值
                x_values[current_y] = x_values[current_y - 1] + (x_values[next_set_y] - x_values[current_y - 1]) /\
                    (next_set_y - current_y + 1)
                current_y += 1

        except StopIteration:
            pass  # Done with lane

        return x_values

    def _lane_points_fit(self, lane):
        # TODO name and docstring
        """ Fits spline in image space for the markers of a single lane (side)

        Parameters
        ----------
        lane: dict as specified in label

        Returns
        -------
        Pixel level values for curve along the y-axis

        Notes
        -----
        This one can be drastically improved. Probably fairly easy as well.
        """
        # NOTE all variable names represent image coordinates, interpolation coordinates are swapped!
        lane = _extend_lane(lane, self.json_content['projection_matrix'])
        sampled_points = self._sample_points(lane, ypp=1)
        self.sampled_points[lane['lane_id']] = sampled_points

        return sampled_points

    def create_all_points(self, ):
        """ Creates splines for given label """
        for lane in self.lanes:
            self._lane_points_fit(lane)


def get_horizontal_values_for_four_lanes(json_path):
    """ Gets an x value for every y coordinate for l1, l0, r0, r1

    This allows to easily train a direct curve approximation. For each value along
    the y-axis, the respective x-values can be compared, e.g. squared distance.
    Missing values are filled with -1. Missing values are values missing from the spline.
    There is no extrapolation to the image start/end (yet).
    But values are interpolated between markers. Space between dashed markers is not missing.

    Parameters
    ----------
    json_path: str
               path to label-file

    Returns
    -------
    List of [l1, l0, r0, r1], each of which represents a list of ints the length of
    the number of vertical pixels of the image
    返回 [l1, l0, r0, r1] 的列表，每个子列表包含整数值，长度与图像的垂直像素数相同

    Notes
    -----
    The points are currently based on the splines. The splines are interpolated based on the
    segmentation values. The spline interpolation has lots of room for improvement, e.g.
    the lines could be interpolated in 3D, a better approach to spline interpolation could
    be used, there is barely any error checking, sometimes the splines oscillate too much.
    This was used for a quick poly-line regression training only.
    当前点基于样条曲线生成。样条插值基于分割值实现，仍有较大改进空间，例如：
    - 可在三维空间进行线段插值
    - 可采用更好的样条插值方法
    - 缺乏错误检查机制
    - 样条曲线有时会出现过度振荡
    本实现仅用于快速折线回归训练
    """

    sc = SplineCreator(json_path)
    sc.create_all_points()

    l1 = sc.sampled_points.get('l1', [-1] * 717)
    l0 = sc.sampled_points.get('l0', [-1] * 717)
    r0 = sc.sampled_points.get('r0', [-1] * 717)
    r1 = sc.sampled_points.get('r1', [-1] * 717)

    lanes = [l1, l0, r0, r1]
    return lanes


def _filter_lanes_by_size(label, min_height=40):
    """ May need some tuning """
    filtered_lanes = []
    for lane in label['lanes']:
        lane_start = min([int(marker['pixel_start']['y']) for marker in lane['markers']])
        lane_end = max([int(marker['pixel_start']['y']) for marker in lane['markers']])
        if (lane_end - lane_start) < min_height:
            continue
        filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _filter_few_markers(label, min_markers=2):
    """Filter lines that consist of only few markers"""
    filtered_lanes = []
    for lane in label['lanes']:
        if len(lane['markers']) >= min_markers:
            filtered_lanes.append(lane)
    label['lanes'] = filtered_lanes


def _fix_lane_names(label):
    """ Given keys ['l3', 'l2', 'l0', 'r0', 'r2'] returns ['l2', 'l1', 'l0', 'r0', 'r1']"""

    # Create mapping
    l_counter = 0
    r_counter = 0
    mapping = {}
    lane_ids = [lane['lane_id'] for lane in label['lanes']]
    for key in sorted(lane_ids): # 按车道线ID排序, l0, l1, l2, l3, r0, r1, r2
        if key[0] == 'l':
            mapping[key] = 'l' + str(l_counter)
            l_counter += 1
        if key[0] == 'r':
            mapping[key] = 'r' + str(r_counter)
            r_counter += 1
    for lane in label['lanes']:
        lane['lane_id'] = mapping[lane['lane_id']]


def read_json(json_path, min_lane_height=20):
    """ Reads and cleans label file information by path"""
    with open(json_path, 'r') as jf:
        label_content = json.load(jf)

    # 过滤高度小于 min_lane_height 的车道线
    _filter_lanes_by_size(label_content, min_height=min_lane_height)
    # 过滤包含少于 min_markers 个标记的车道线
    _filter_few_markers(label_content, min_markers=2)
    # 修复车道线名称
    _fix_lane_names(label_content)

    content = {'projection_matrix': label_content['projection_matrix'], 'lanes': label_content['lanes']}

    # 坐标值数据类型变换, str -> int/float
    for lane in content['lanes']:
        for marker in lane['markers']:
            for pixel_key in marker['pixel_start'].keys():
                marker['pixel_start'][pixel_key] = int(marker['pixel_start'][pixel_key])
            for pixel_key in marker['pixel_end'].keys():
                marker['pixel_end'][pixel_key] = int(marker['pixel_end'][pixel_key])
            for pixel_key in marker['world_start'].keys():
                marker['world_start'][pixel_key] = float(marker['world_start'][pixel_key])
            for pixel_key in marker['world_end'].keys():
                marker['world_end'][pixel_key] = float(marker['world_end'][pixel_key])
    return content


def ir(some_value):
    """ Rounds and casts to int
    Useful for pixel values that cannot be floats
    Parameters
    ----------
    some_value : float
                 numeric value
    Returns
    --------
    Rounded integer
    Raises
    ------
    ValueError for non scalar types
    """
    return int(round(some_value))


# End code under the previous license
