# DVMPC
#Deep Visual MPC-Policy Learning for Navigation

 
**Summary**: VUNet can predict the future image from current and previous image view and virtual velocities. Our method can take into account for

**I.** Robot pose change, and
**II.** Dynamic object (For example Pedestrians).   

By feeding predicted images into our previous work, GONet, the robot can understand when and where the robot collide with the obstacle only from an RGB Camera view.
Please see the [website](http://svl.stanford.edu/projects/vunet/) (http://svl.stanford.edu/projects/vunet/) for more technical details. This repository is intended for distribution of the code and its instruction.

#### Paper
**["VUNet: Dynamic Scene View Synthesis for Traversability Estimation using an RGB Camera"](https://ieeexplore.ieee.org/document/8624332)**


System Requirement
=================
Ubuntu 16.04

Chainer 4.1.0

Python Pillow 1.1.7

ROS KINETIC(http://wiki.ros.org/kinetic)

Nvidia GPU

How to use VUNet
=================

We are providing VUNet with GONet, which can realize the early obstacle detection.
By giving the multiple virtual velocities, it is easy to implement another application example, multi-path traversability estimation.

#### Step1: Download
git clone https://github.com/NHirose/VUNet.git

#### Step2: Camera Setup
VUNet can only accept the fish eye camera image to capture the environment in front of the robot.
We highly recommend to use RICOH THETA S, because the training and the evaluation of VUNet are done by the collected images by RICOH THETA S.(https://theta360.com/en/about/theta/s.html)
Please put the camera in front of your device(robot) at the height 0.460 m not to caputure your device itself and connect with your PC by USB cable.

#### Step3: Image Capturing
To turn on RICOH THETA S as the live streaming mode, please hold the bottom buttom at side for about 5 senconds and push the top buttom.(Detail is shown in the instrunction sheet of RICOH THETA S.)

To capture the image from RICOH THETA S, we used the open source in ROS, cv_camera_node(http://wiki.ros.org/cv_camera).
The subscribed topic name of the image is "/cv_camera_node1/image1". We recommend that the flame rate of image is 3 fps.

#### Step4: Subscribing velocities
The application of early obstacle detection needs to subscribe the robot and tele-operator's velocities.
You can use the tele-operator's velocity at the previous step instead of the robot velocity, if you can not take the robot odometry.
The subscribed topic name of the robot odometry and the tele-operator's velocity are "/odom" and "/cmd_vel_mux/input/teleop".


#### Step5: Runing VUNet
The last process to get the traversable probability is just to run our algorithm.

python VUNet_earlyobstacle_detection.py

The published topic name for the traversable probability is "img_all".
In this code, we subscribe current, and the predicted images for 4 steps with the traversable probability.

License
=================
The codes provided on this page are published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License(https://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact us for further options. 

Citation
=================

If you use VUNet's software or database, please cite:
```
@article{hirose2019vunet,
  title={VUNet: Dynamic Scene View Synthesis for Traversability Estimation using an RGB Camera},
  author={Hirose, Noriaki and Sadeghian, Amir and Xia, Fei and Mart{\'\i}n-Mart{\'\i}n, Roberto and Savarese, Silvio},
  journal={IEEE Robotics and Automation Letters},
  year={2019},
  publisher={IEEE}
}

```
or
```
@article{hirose2018gonet++,
  title={Gonet++: Traversability estimation via dynamic scene view synthesis},
  author={Hirose, Noriaki and Sadeghian, Amir and Xia, Fei and Savarese, Silvio},
  journal={arXiv preprint arXiv:1806.08864},
  year={2018}
}

```
