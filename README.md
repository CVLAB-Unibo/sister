# SiSter: Single camera Stereo Robot

Original paper: *Beyond the Baseline: 3D Reconstructionof Tiny Objects with Single CameraStereo Robot*

## Table of contents

* [Dataset](#dataset)
* [C++ Implementation](#cpp)
* [ROS Implementation](#ros)


<a name="dataset" />

# Sister Dataset

[Download Dataset](http://github.com)

The dataset contains the following objects:

```
arduino
component_0J
component_1B
component_1G
hexa_nut
hexa_screw
nodemcu
washer
```

Each subfolder image has this naming convention:

```
<OBJECT_NAME> / <CAMERA_DISTANCE> / <BASELINE> / <center|left|top|right|bottom>.png
```

for example:

```
component_1G/10cm/025mm/left.png
```

means the `left` view of the rig with a baseline of `25mm` with a camera-to-object distance of `10cm`, of the component with name `component_1G`.

Each groundtruth depth image, instead, has a similar naming convention:

```
<OBJECT_NAME> / <CAMERA_DISTANCE> / gt_depth.exr
```

given that the depth groundtruth is associated with the center image of each rig and it varies with the camera distance (but not with the rig baseline).  

<a name="cpp" />

# Sister C++: 5 views disparity computation

This repository containes a prototype implementation of the multiview stereo algorithm described in the paper. The implementations is contained in the `cpp` folder and is basically wrapped around a single helper class.

#### Requirements

```
OpenCV 
Eiegen3
OpenMP
```

#### Compile

```
cd cpp
mkdir build
cd build
cmake ..
make
```

#### Run example

The above compile command will build the library along with a sample application `compute_disp` that will load 5 images and calculate multiview disparity. You can run the example on the sister dataset folders themselves. For example you can run

```
/compute_disp $SISTER_DATASET/component_1G/10cm/025mm/ 192
```

where `$SISTER_DATASET` is the root folder of the dataset described above. The second argument is the full path of the folder containing the 5 images (pay attention to naming convention, images names should be `<center|left|top|right|bottom>.png`). The second argument `192` is the max disparity value.

<a name="ros" />

# ROS implementation

[ROS Example](https://github.com/CVLAB-Unibo/sister/tree/f9044bbdadbe303b71c40f49f3c68de9a9ec5d64/ros)
