#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import json


APOLLO_SYN_IMG_SIZE = (1080, 1920)
TUSIMPLE_IMG_SIZE = (720, 1280)

# Gray
IMAGINARY_LANE_COLOR = (128, 128, 128)

LANE_COLORS = {
    "l3": (255, 128, 50),  # 
    "l2": (255, 128, 0),  # 
    "l1": (255, 0, 128),  # green
    "l0": (255, 0, 0),  # blue
    "pad": (128, 128, 128),  # gray
    "r0": (0, 0, 255),  # light red
    "r1": (0, 128, 255),  # light green
    "r2": (128, 0, 255),  # light yellow
    "r3": (50, 128, 255),  # light cyan
}

MARKER_COLORS = [
    (0, 0, 255),  # blue
    (0, 255, 0),  # green
    (255, 0, 0),  # red
    (0, 0, 128),  # cyan
    (0, 128, 0),  # yellow
    (128, 0, 0),  # magenta
    (255, 255, 0),  # orange
    (255, 0, 255),  # pink
    (0, 255, 255),  # cyan
    (128, 128, 0),  # olive
    (128, 0, 128),  # purple
    (0, 128, 128),  # teal
    (100, 0, 0),  # brown
    (0, 100, 0),  # green
    (0, 0, 100),  # blue
    (100, 100, 0),  # yellow
    (100, 0, 100),  # magenta
    (0, 100, 100),  # cyan
]


def convert_image(old_image_path, new_image_path):
    image = cv2.imread(old_image_path)
    assert image.shape == (APOLLO_SYN_IMG_SIZE[0], APOLLO_SYN_IMG_SIZE[1], 3)
    image = cv2.resize(image, (TUSIMPLE_IMG_SIZE[1], TUSIMPLE_IMG_SIZE[0]))
    cv2.imwrite(new_image_path, image)


def norm_uv_to_tusimple_uv(norm_u, norm_v):
    u = int(round(norm_u * TUSIMPLE_IMG_SIZE[1]))
    v = int(round(norm_v * TUSIMPLE_IMG_SIZE[0]))
    return [u, v]

def marker_is_almost_horizontal_or_vertical(marker_points,
                                            max_k=5, #3, # 水平
                                            min_k=0.000001, #0.01 # 垂直
                                            ):
    # 直线拟合marker点，得到斜率
    # marker_points: (u, v)
    # 返回斜率k, 截距b
    if len(marker_points) < 2:
        #print(f"marker_points must have at least 2 points")
        return False, 0, 0
    if len(marker_points) == 2:
        return marker_points[0][1] == marker_points[1][1] or marker_points[0][0] == marker_points[1][0], 0, 0
    
    new_marker_points = np.array(marker_points)
    u = new_marker_points[:, 0]
    v = new_marker_points[:, 1]
    min_v = min(v)
    max_v = max(v)
    min_u = min(u)
    max_u = max(u)
    if max_v - min_v < max_u - min_u:
        k, b = np.polyfit(u, v, 1)
        k = -1.0 / (k + 1e-6)
    else:
        k, b = np.polyfit(v, u, 1)
    #print(f"k: {k}, b: {b}")
    if abs(k) < min_k:
        return True, k, b
    if abs(k) > max_k:
        return True, k, b
    return False, k, b


def convert_lane_annotation(
    old_lane_annotation_path, new_lane_annotation_path, new_image_path="",
    min_lane_height=1, #20,
    min_marker_points=2, #2,
    filter_same_v=False,
    filter_imaginary=True,
):
    tusimple_anno = {
        "lanes": [],
        "h_samples": list(range(160, 720, 10)),
        "raw_file": new_image_path,
    }
    with open(old_lane_annotation_path, "r") as f:
        lines = f.readlines()
    lanes = {
        "l1": {},  # { '<marker_id>': {'points': [...], 'type': '', 'topology_type': ''}, ... }
        "l0": {},
        "r0": {},
        "r1": {},
    }
    for line in lines:
        # Example: 3686 Imaginary 0.028 0.919 -1 MergeLaneRight White -3.000 1.497 6.663
        # 3686: lane id
        # Imaginary: lane type
        # 0.028: normalized lane point u
        # 0.919: normalized lane point v
        # -1: ego-centric lane index, -4,-3,-2,-1, 1, 2, 3, 4 for l3, l2, l1, l0, r0, r1, r2, r3
        # MergeLaneRight: lane topology type
        # White: lane color
        # -3.000 1.497 6.663: 3D position of this sample in camera coordinate
        point_info = line.split(" ")
        assert len(point_info) == 10
        # (SingleSolid, SingleDash, DoubleSolid, DoubleDash, LeftDashRightSolid, LeftSolidRightDash, Curb, Imaginary, Other)
        marker_id = point_info[0]  # digit string
        lane_type = point_info[1]
        norm_u = float(point_info[2])  # u, 0-1
        norm_v = float(point_info[3])  # v, 0-1
        ego_lane_idx = int(point_info[4])

        # MergeLaneLeft, MergeLaneRight, SplitLaneLeft, SplitLaneRightForkLaneLeft, ForkLaneRight, MergeLaneLeft, MergeLaneRight, ParkingLane, CenterLane
        lane_topology_type = point_info[5]

        #if ego_lane_idx < 0 and lane_topology_type == "MergeLaneLeft":
        #    continue
        #if ego_lane_idx > 0 and lane_topology_type == "MergeLaneRight":
        #    continue

        if ego_lane_idx == -2:
            lane_id = "l1"
        elif ego_lane_idx == -1:
            lane_id = "l0"
        elif ego_lane_idx == 1:
            lane_id = "r0"
        elif ego_lane_idx == 2:
            lane_id = "r1"
        else:
            continue
        #print(line)

        if marker_id not in lanes[lane_id]:
            lanes[lane_id][marker_id] = {
                "points": [],
                "type": lane_type,
                "topology_type": lane_topology_type,
                "k": 0,
                "filtered": False,
            }
        lanes[lane_id][marker_id]["points"].append(norm_uv_to_tusimple_uv(norm_u, norm_v))

        
    # Sort marker points ascending by v
    for lane_id in lanes.keys():
        for marker_id in lanes[lane_id].keys():
            lanes[lane_id][marker_id]["points"].sort(key=lambda x: x[1])

    # remove the same v in the same marker
    if filter_same_v:
        for lane_id in lanes.keys():
            for marker_id in lanes[lane_id].keys():
                points = lanes[lane_id][marker_id]["points"]
            new_points = []
            for pt in points:
                if len(new_points) == 0:
                    new_points.append(pt)
                    continue
                if pt[1] != new_points[-1][1]:
                    new_points.append(pt)
            lanes[lane_id][marker_id]["points"] = new_points

    # merge lanes with the same ego_lane_idx and lane_topology_type
    merged_lanes = {}
    for lane_id in lanes.keys():
        new_marker_dict = {}
        for marker_id in lanes[lane_id].keys():
            #print(f"Marker {marker_id}")
            # Remove lane marker that has less than 2 points
            marker = lanes[lane_id][marker_id]
            points_len = len(marker["points"])
            # Filter imaginary lane
            if filter_imaginary:
                if marker["type"] == "Imaginary":
                    marker["filtered"] = True
                    continue
            # Filter lane marker that has less than 2 points
            if points_len < min_marker_points:
                print(f"Remove marker {marker_id} from lane {lane_id} because it has {points_len} points, less than {min_marker_points} ")
                marker["filtered"] = True
                continue
            # Remove lane marker that is almost horizontal or vertical
            filtered, k , b = marker_is_almost_horizontal_or_vertical(lanes[lane_id][marker_id]["points"])
            if filtered:
                print(f"Remove marker {marker_id} from lane {lane_id} because it is almost horizontal or vertical")
                marker["k"] = k
                marker["filtered"] = True
                continue
            else:
                marker["k"] = k

            # Filter marker by topology type
            #if marker["topology_type"] == "MergeLaneLeft" and lane_id.startswith("l"):
            #    marker["filtered"] = True
            #if marker["topology_type"] == "MergeLaneRight" and lane_id.startswith("r"):
            #    marker["filtered"] = True
            
                
            # Merge marker by topology type
            new_marker_id = f"{lane_id}-{marker['topology_type']}"
            if new_marker_id not in new_marker_dict:
                new_marker_dict[new_marker_id] = {
                    "points": [],
                    "type": marker["type"],
                    "topology_type": marker["topology_type"],
                }
            new_marker_dict[new_marker_id]["points"].extend(marker["points"])
        merged_lanes[lane_id+"-merged"] = new_marker_dict

    # Sort marker points ascending by v
    for lane_id in merged_lanes.keys():
        for marker_id in merged_lanes[lane_id].keys():
            merged_lanes[lane_id][marker_id]["points"].sort(key=lambda x: x[1])        
                
    # Remove lane that covers less than 20 pixels            
    for lane_id in merged_lanes.keys():
        lane = merged_lanes[lane_id]
        min_v = APOLLO_SYN_IMG_SIZE[0]
        max_v = 0
        if len(lane.keys()) == 0:
            continue
        for marker_id in lane.keys():
            if len(lane[marker_id]["points"]) == 0:
                continue
            min_v = min(min_v, lane[marker_id]["points"][0][1])
            max_v = max(max_v, lane[marker_id]["points"][-1][1])
        if max_v - min_v < min_lane_height:
            print(f"Remove lane {lane_id} because it covers less than {min_lane_height} pixels")
            merged_lanes[lane_id] = {}

    tusimple_anno["old_lanes"] = lanes
    tusimple_anno["merged_lanes"] = merged_lanes
    # Write to new lane annotation file
    with open(new_lane_annotation_path, "w") as f:
        json.dump(tusimple_anno, f)

    return tusimple_anno

def lane_id2width(lane_id):
    if lane_id == "l1" or lane_id == "l1-merged":
        return 4
    elif lane_id == "l0" or lane_id == "l0-merged":
        return 3
    elif lane_id == "r0" or lane_id == "r0-merged":
        return 2
    elif lane_id == "r1" or lane_id == "r1-merged":
        return 1
    else:
        raise ValueError(f"Invalid lane id: {lane_id}")

def get_laneidx_by_name(lane_name):
    if lane_name == "l3" or lane_name == "l3-merged":
        return -4
    elif lane_name == "l2" or lane_name == "l2-merged":
        return -3
    elif lane_name == "l1" or lane_name == "l1-merged":
        return -2
    elif lane_name == "l0" or lane_name == "l0-merged":
        return -1
    elif lane_name == "r0" or lane_name == "r0-merged":
        return 1
    elif lane_name == "r1" or lane_name == "r1-merged":
        return 2
    elif lane_name == "r2" or lane_name == "r2-merged":
        return 3
    elif lane_name == "r3" or lane_name == "r3-merged":
        return 4
    else:
        raise ValueError(f"Invalid lane name: {lane_name}")

def get_lanepx_color_by_ego_idx(ego_idx, lane_type=""):
    ego_idx = int(ego_idx) + 4
    ego_idx_str = list(LANE_COLORS.keys())[ego_idx]
    return LANE_COLORS[ego_idx_str]

def draw_image_apollo_anno(image_path, apollo_anno_path, output_path):
    image = cv2.imread(image_path)
    with open(apollo_anno_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        point_info = line.split(" ")
        assert len(point_info) == 10
        marker_id = point_info[0]
        lane_type = point_info[1]
        norm_u = float(point_info[2])
        norm_v = float(point_info[3])
        ego_lane_idx = int(point_info[4])

        u = int(round(norm_u * APOLLO_SYN_IMG_SIZE[1]))
        v = int(round(norm_v * APOLLO_SYN_IMG_SIZE[0]))
        thickness = 1 if lane_type == "Imaginary" else -1
        cv2.circle(image, (u, v), 5, get_lanepx_color_by_ego_idx(ego_lane_idx, lane_type), thickness)
    cv2.putText(image, image_path, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(output_path, image)
    return image

def draw_image_w_old_lanes(image_path, tusimple_anno, output_path, color_by_marker=False):
    image = cv2.imread(image_path)
    marker_cnt = 0
    for lane_id in tusimple_anno["old_lanes"].keys():
        lane = tusimple_anno["old_lanes"][lane_id]
        for marker_id in lane.keys():
            if color_by_marker:
                color = MARKER_COLORS[marker_cnt % len(MARKER_COLORS)]
                marker_cnt += 1
            else:
                try:
                    if lane[marker_id]["filtered"]:
                        color = IMAGINARY_LANE_COLOR
                    else:
                        color = get_lanepx_color_by_ego_idx(get_laneidx_by_name(lane_id), lane[marker_id]["type"])
                    marker_cnt += 1
                except KeyError:
                    print(lane[marker_id])
                    raise KeyError(f"Invalid k of lane id: {lane_id}")
            #for u, v in tusimple_anno["old_lanes"][lane_id][marker_id]["points"]:
            for pt_idx in range(len(lane[marker_id]["points"][:-1])):
                u, v = lane[marker_id]["points"][pt_idx]
                u_next, v_next = lane[marker_id]["points"][pt_idx + 1]
                thickness = 1 if lane[marker_id]["type"] == "Imaginary" else -1
                cv2.circle(image, (u, v), 8, color, thickness)
                cv2.line(image, (u, v), (u_next, v_next), color, lane_id2width(lane_id))
    # display image_path on the top left of the image
    cv2.putText(image, image_path, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(output_path, image)
    return image



def draw_apollo_anno(anno_dir, output_video=True, outv=None, outv_org=None):
    first_layer = True if output_video and outv is None else False
    if output_video and outv is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outv = cv2.VideoWriter('apollo-output-merged-by-ego-lane-idx-and-lane-topology-type2.mp4', fourcc, 1.0, (TUSIMPLE_IMG_SIZE[1], TUSIMPLE_IMG_SIZE[0]))
        #outv_org = cv2.VideoWriter('apollo-output-org.mp4', fourcc, 1.0, (APOLLO_SYN_IMG_SIZE[1], APOLLO_SYN_IMG_SIZE[0]))
    for anno_file in os.listdir(anno_dir):
        anno_path = os.path.join(anno_dir, anno_file)
        if os.path.isfile(anno_path) and anno_file.endswith(".txt"):
            # Find a annotation file
            tusimple_anno_path = anno_path.replace(".txt", ".json")
            old_img_path = anno_path.replace(".txt", ".jpg").replace("LaneLine_GroundTruth", "RGB")
            assert os.path.exists(old_img_path)
            new_img_path = old_img_path.replace(".jpg", "_tusimple.jpg")
            apollo_labeled_img_path = old_img_path.replace(".jpg", "_apollo_labeled.jpg")
            org_labeled_img_path = old_img_path.replace(".jpg", "_apollo_org_labeled.jpg")
            print(f"Convert {old_img_path} to {new_img_path}")
            convert_image(old_img_path, new_img_path)
            ts_anno = convert_lane_annotation(
                anno_path, tusimple_anno_path, new_img_path
            )            
            out_img = draw_image_w_old_lanes(new_img_path, ts_anno, apollo_labeled_img_path)   
            out_img_org = draw_image_apollo_anno(old_img_path, anno_path, org_labeled_img_path)
            if output_video and outv is not None:
                outv.write(out_img)
            if output_video and outv_org is not None:
                outv_org.write(out_img_org)
        elif os.path.isdir(anno_path):
            if anno_file == "SEVERE_DEGRADATION":
                continue
            if anno_file == "DEGRADATION":
                continue
            draw_apollo_anno(anno_path, output_video, outv, outv_org)
    if output_video and first_layer:
        outv.release()
        print("Output video to apollo-output-merged-by-ego-lane-idx-and-lane-topology-type2.mp4")
    if outv_org and first_layer:
        outv_org.release()
        print("Output video to apollo-output-org.mp4")


def test_one():
    test_image_path = "datasets/apollo/RGB/13-00/CLEAR_SKY/NO_DEGRADATION/With_Pedestrian/With_TrafficBarrier/Residential/Traffic_027/0000001.jpg"
    test_lane_annotation_path = "datasets/apollo/LaneLine_GroundTruth/13-00/CLEAR_SKY/NO_DEGRADATION/With_Pedestrian/With_TrafficBarrier/Residential/Traffic_027/0000001.txt"
    new_image_path = "apollo2tusimple_test.jpg"
    convert_image(test_image_path, new_image_path)
    ts_anno = convert_lane_annotation(
        test_lane_annotation_path, "test.json", new_image_path
    )
    draw_image_w_old_lanes(new_image_path, ts_anno)
    # convert_lane_annotation(test_lane_annotation_path, 'test.txt')


if __name__ == "__main__":
    #test_one()
    draw_apollo_anno("datasets/apollo/LaneLine_GroundTruth")
