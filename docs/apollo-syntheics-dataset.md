# Apollo Syntheics Dataset

[Project Website](https://developer.apollo.auto/synthetic.html)

## Overview
Apollo Synthetic 是一个用于自动驾驶的照片级逼真合成数据集。它包含 273,000 个不同的（非连续的视频帧）帧，这些帧来自各种高保真度的虚拟场景，包括高速公路、城市、住宅区、市中心和室内停车场环境。这些虚拟世界是使用 Unity 3D 引擎创建的。合成数据集的最大优势在于它提供精确的地面实况数据。另一个好处是它能够提供更多的环境变化（这在现实世界中相对更难实现且成本更高），例如一天中的不同时间、不同的天气条件、不同的交通/障碍物以及不同的路面质量。我们的数据集提供了广泛的地面实况数据：2D/3D 对象数据、语义/实例级分割、深度数据和 3D 车道线数据。

- 图像分辨率： **1920 * 1080**

## Directoies

- 车道线标签路径: `LaneLine_GroundTruth/00-00/CLEAR_SKY/DEGRADATION/With_Pedestrian/With_TrafficBarrier/Residential/Traffic_003/0000000.txt`
- RGB图像: `RGB/00-00/CLEAR_SKY/DEGRADATION/With_Pedestrian/With_TrafficBarrier/Residential/Traffic_003/0000000.jpg`

## Lane Annotation Format

每条车道线的可见部分沿其三维长度（每1米）定期采样，并以点序列的形式输出。这些线代表了从“自我”视角来看车道线的**内边界**。对于每条线，其样本按“自我”行进的方向逐行列出。**每行代表一个点**，包含以下以空格分隔的字段：
```
    1 global id
    1 lane marker type (SingleSolid, SingleDash, DoubleSolid, DoubleDash, LeftDashRightSolid, LeftSolidRightDash, Curb, Imaginary, Other)
    2 normalized pixel position of this lane point sample (the origin at top-left)
    1 ego-centric lane index (-4 ~ 4, e.g. -1 means the left boundary of the ego lane and 1 means the right boundary of the ego lane.)
    1 lane topology type (ForkLaneLeft, ForkLaneRight, MergeLaneLeft, MergeLaneRight, ParkingLane, CenterLane)
    1 lane marker color (White, Yellow)
    3 3D position of this sample in camera coordinate
```
Example:
```
3686 Imaginary 0.028 0.919 -1 MergeLaneRight White -3.000 1.497 6.663
3686 Imaginary 0.037 0.857 -1 MergeLaneRight White -3.357 1.456 7.596
3686 Imaginary 0.037 0.810 -1 MergeLaneRight White -3.757 1.416 8.512
3686 Imaginary 0.032 0.773 -1 MergeLaneRight White -4.201 1.377 9.407
3686 Imaginary 0.022 0.743 -1 MergeLaneRight White -4.686 1.339 10.280
3686 Imaginary 0.009 0.718 -1 MergeLaneRight White -5.212 1.302 11.130
489 SingleSolid 0.424 0.476 1 MergeLaneLeft White -4.198 -0.753 58.186
489 SingleSolid 0.408 0.476 1 MergeLaneLeft White -5.088 -0.733 57.731
489 SingleSolid 0.391 0.477 1 MergeLaneLeft White -5.979 -0.713 57.278
489 SingleSolid 0.373 0.477 1 MergeLaneLeft White -6.871 -0.693 56.827
5867 Imaginary 0.355 0.478 1 MergeLaneLeft White -7.792 -0.676 56.432
5867 Imaginary 0.337 0.478 1 MergeLaneLeft White -8.685 -0.657 55.983
5867 Imaginary 0.319 0.479 1 MergeLaneLeft White -9.579 -0.637 55.535
5867 Imaginary 0.301 0.479 1 MergeLaneLeft White -10.474 -0.618 55.089
5867 Imaginary 0.282 0.480 1 MergeLaneLeft White -11.369 -0.598 54.643
5867 Imaginary 0.263 0.480 1 MergeLaneLeft White -12.263 -0.579 54.196
5867 Imaginary 0.243 0.481 1 MergeLaneLeft White -13.158 -0.559 53.749
467 SingleSolid 0.224 0.481 1 MergeLaneLeft White -14.024 -0.537 53.246
467 SingleSolid 0.204 0.482 1 MergeLaneLeft White -14.920 -0.518 52.802
467 SingleSolid 0.183 0.482 1 MergeLaneLeft White -15.819 -0.499 52.366
445 SingleSolid 0.162 0.483 1 MergeLaneLeft White -16.723 -0.480 51.938
445 SingleSolid 0.141 0.483 1 MergeLaneLeft White -17.630 -0.462 51.517
445 SingleSolid 0.120 0.484 1 MergeLaneLeft White -18.541 -0.444 51.105
445 SingleSolid 0.098 0.484 1 MergeLaneLeft White -19.456 -0.426 50.702
445 SingleSolid 0.075 0.485 1 MergeLaneLeft White -20.374 -0.409 50.306
445 SingleSolid 0.053 0.485 1 MergeLaneLeft White -21.296 -0.392 49.918
445 SingleSolid 0.030 0.486 1 MergeLaneLeft White -22.221 -0.375 49.539
445 SingleSolid 0.006 0.486 1 MergeLaneLeft White -23.150 -0.359 49.168
3687 Imaginary 0.672 0.999 1 MergeLaneRight White 0.940 1.537 5.737
3687 Imaginary 0.605 0.916 1 MergeLaneRight White 0.671 1.495 6.699
3687 Imaginary 0.550 0.854 1 MergeLaneRight White 0.363 1.454 7.650
3687 Imaginary 0.502 0.807 1 MergeLaneRight White 0.016 1.413 8.587
3687 Imaginary 0.459 0.769 1 MergeLaneRight White -0.368 1.373 9.509
3687 Imaginary 0.421 0.739 1 MergeLaneRight White -0.789 1.333 10.415
3687 Imaginary 0.384 0.713 1 MergeLaneRight White -1.247 1.294 11.303
3687 Imaginary 0.350 0.692 1 MergeLaneRight White -1.740 1.256 12.172
3687 Imaginary 0.317 0.675 1 MergeLaneRight White -2.268 1.219 13.020
3687 Imaginary 0.286 0.659 1 MergeLaneRight White -2.830 1.183 13.847
3687 Imaginary 0.255 0.646 1 MergeLaneRight White -3.425 1.148 14.649
3687 Imaginary 0.226 0.634 1 MergeLaneRight White -4.041 1.114 15.437
3687 Imaginary 0.198 0.624 1 MergeLaneRight White -4.671 1.080 16.213
3687 Imaginary 0.172 0.615 1 MergeLaneRight White -5.314 1.046 16.978
3687 Imaginary 0.147 0.607 1 MergeLaneRight White -5.971 1.014 17.731
3687 Imaginary 0.123 0.599 1 MergeLaneRight White -6.641 0.981 18.473
3687 Imaginary 0.100 0.592 1 MergeLaneRight White -7.324 0.949 19.202
3687 Imaginary 0.078 0.586 1 MergeLaneRight White -8.020 0.918 19.920
3687 Imaginary 0.056 0.580 1 MergeLaneRight White -8.728 0.887 20.625
3687 Imaginary 0.035 0.575 1 MergeLaneRight White -9.449 0.857 21.317
3687 Imaginary 0.015 0.570 1 MergeLaneRight White -10.182 0.827 21.997
```


## 转换到tusimple的规则
- 图像尺寸resize到1280*720;
- 车道线标签只取lane index: [-2, -1, 1, 2]4条，允许空车道线；
- 对于车道线区域，进行线性插值；
- 按照tusimple的h_samples进行采样；
- 以图片为单位，按照0.8,0.1,0.1的比例分割为train/valid/test数据集；

