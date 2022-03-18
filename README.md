# VAMR_project

### Computer Specifications

* Processor: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz
* RAM: 16.0GB
* Graphic Card: Nvidia GeForce RTX 2060 Labtop 
* Operating System: 64-bit Windows 10 Pro

### Built With

* MATLAB version R2021b
* Library: Computer Vision Toolbox, Image Processing Toolbox

<!-- HOW TO USE -->
## Getting Started

1. Place KITTI, Malaga, Parking datasets(images) in the following location from the current folder, where _project_main.m_ is located.
* KITTI: `../data/kitti05/kitti`
* Malaga: `../data/malaga-urban-dataset-extract-07`
* Parking: `../data/parking`
* The path can be edited from the file _get_ds_vars.m_

2. From _project_main.m_, set up `video_file`, `ds`, and `ba_bool`
* `video_file` is the name of the result video which will be stored in the current folder.
* `ds` is the testing dataset (0: KITTI, 1: Malaga, 2: parking).
* `ba_bool` is the bundle adjustment boolean variable. Set true if you want to use the bundle adjustment.

3. Run _project_main.m_. The generated video will be automatically saved in the current folder.

## Function Details

* _detectHarrisFeatures_: Detect harris corners
* _vision.PointTracker_: Track points from one frame to the next using KLT
* _estimateFundamentalMatrix_: Estimate fundamental matrix using 8-point ransac
* _relativeCameraPose_: Recover the camera pose from the fundamental matrix
* _estimateWorldCameraPose_: Estiame pose from 2D-3D correspondences using P3P ransac
* _worldToImage_: Project 3D landmarks back into image plane
* _triangulate_: Triangulate 3D landmarks using image points and poses of two frames
* _budleAdjustment_: Pose and 3D landmarks Refinement
* _bundleAdjustmentMotion_: Pose refinement while keeping the 3D landmarks fixed

## Results
Please visit the following [link](https://www.youtube.com/playlist?list=PL41a83_sExo0vBHgH7e4u8pi1S1inXTLd) for the evaluation results on KITTI-05, Malaga and Parking Datasets

<!-- CONTACT -->
## Contact

* Mantan Patel  - patelm@student.ethz.ch
* Shobhit Singhal - ssinghal@student.ethz.ch
* Manuel Grossenbacher - grmanuel@student.ethz.ch
* Jinhoo Kim - kimjin@student.ethz.ch



<p align="right">(<a href="#top">back to top</a>)</p>


