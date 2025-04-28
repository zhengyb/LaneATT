# 统计openlanev1数据集的场景数量

import os
import json


def count_scenes(scene_anno_file):
    print(f"scene_anno_file: {scene_anno_file}")
    with open(scene_anno_file, 'r') as f:
        scenes = json.load(f)
        print(f"scenes: {len(scenes)}")
    scene_count = {}
    weather_count = {}
    hours_count = {}
    for key, value in scenes.items():
        """
        "segment-17759280403078053118_6060_580_6080_580_with_camera_labels": {
            "scene": "Suburbs",
            "weather": "Clear",
            "hours": "Daytime"
        },  
        """
        if key == 'segment-17759280403078053118_6060_580_6080_580_with_camera_labels':
            print(value)
        scene = value['scene']
        weather = value['weather']
        hours = value['hours']
        scene_count[scene] = scene_count.get(scene, 0) + 1
        weather_count[weather] = weather_count.get(weather, 0) + 1
        hours_count[hours] = hours_count.get(hours, 0) + 1

    scene_sum = sum(scene_count.values())
    weather_sum = sum(weather_count.values())
    hours_sum = sum(hours_count.values())
    scene_count['sum'] = scene_sum
    weather_count['sum'] = weather_sum
    hours_count['sum'] = hours_sum

    print(f"scene_count: {scene_count}")
    print(f"weather_count: {weather_count}")
    print(f"hours_count: {hours_count}")

    return scene_count, weather_count, hours_count


def main():
    scene_anno_file = 'datasets/openlane/scene.json'
    count_scenes(scene_anno_file)


if __name__ == "__main__":
    main()
