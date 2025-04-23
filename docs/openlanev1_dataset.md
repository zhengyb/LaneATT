# Openlane V1.x Lane Dataset


Details of Openlane v1 lane annotation, refer to [here](https://github.com/OpenDriveLab/OpenLane/blob/main/anno_criterion/Lane/README.md)

- 10 images per second.
- 1920 * 1280
- 'attribute' might be wrong.
- Lanes might be splited into two or more items due to the category. In our case, it is better to merge them. But how?


## Directoies
TODO

## Annotation Format

Here, we only explain the data of 2D lanes.
```JSON
{
    "lane_lines": [                         (k lanes in `lane_lines` list)
        {
            "category":                     <int> -- lane category
                                                        0: 'unkown',
                                                        1: 'white-dash',
                                                        2: 'white-solid',
                                                        3: 'double-white-dash',
                                                        4: 'double-white-solid',
                                                        5: 'white-ldash-rsolid',
                                                        6: 'white-lsolid-rdash',
                                                        7: 'yellow-dash',
                                                        8: 'yellow-solid',
                                                        9: 'double-yellow-dash',
                                                        10: 'double-yellow-solid',
                                                        11: 'yellow-ldash-rsolid',
                                                        12: 'yellow-lsolid-rdash',
                                                        20: 'left-curbside',
                                                        21: 'right-curbside'
            "visibility":                   <float> [n, ] -- visibility of each point
            "uv":[                          <float> [2, n] -- 2d lane points under image coordinate
                [u1,u2,u3...],
                [v1,v2,v3...]
            ],
            "attribute":                    <int> -- left-right attribute of the lane
                                                        1: left-left
                                                        2: left
                                                        3: right
                                                        4: right-right
        },
        ...
    ],
    "file_path":                            <str> -- image path
}
```


## Convertation
### 20250423
- 图片采样率：1/5
- 图像处理: 
  - 保持长宽比不变, resize为1280宽度；
  - 截去图像顶部，变为720高度；
- Lane label处理
  - 只在2D标注车道线处理；
  - 只取attribute标注为1-4的车道线；
  - 根据图像处理方法变换车道线像素点坐标；
  - 相同y的坐标点，x取平均值；
  - 对车道线在y轴方向上的空洞点，作线性插值；
  - 按tusimple的h_samples采样坐标点；
  - 过滤点标注点少于2的车道线；
- 存在的问题：
  - 上述处理后，会过滤掉不少点数不达标的车道线。
  - attribute标注可能不准；
  - 因为openlane区分车道线类别，在空间上属于同一条车道线的坐标点会被分割为不同的车道线。比如，实线与虚线相连，会标注成2条线。在我们的项目中应该合并。
  - openlane会标注“路边”, 如果取全部车道线，会存在车道线与路边两条lane对象很靠近。应该如何处理？