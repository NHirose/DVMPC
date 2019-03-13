# Deep Visual MPC-Policy Learning for Navigation
 
**Summary**: Our method, DVMPC can realize the navigation with obstacle avoidance by only using an RGB image. Our control policy, PoliNet is trained under the same objectives as Model Predictive Control(MPC), which includes, image loss, traversability loss and reference loss.

Please see the [website](http://svl.stanford.edu/projects/dvmpc/) (http://svl.stanford.edu/projects/dvmpc/) for more technical details. This repository is intended for distribution of the code and its instruction.

#### Paper
**["Deep Visual MPC-Policy Learning for Navigation"](https://arxiv.org/abs/1903.02749)**


System Requirement
=================
Ubuntu 16.04

Chainer 4.1.0

Python Pillow 1.1.7

ROS KINETIC(http://wiki.ros.org/kinetic)

Nvidia GPU

How to use DVMPC
=================

We are providing DVMPC, which can realize the navigation with obstacle avoidance by only using an RGB image.

#### Step1: Download
git clone https://github.com/NHirose/DVMPC.git

#### Step2: Camera Setup
DVMPC can only accept the 360-degree camera image to capture the environment in front of the robot.
We highly recommend to use RICOH THETA S.(https://theta360.com/en/about/theta/s.html)
Please put the camera in front of your device(robot) at the height 0.460 m not to caputure your device itself and connect with your PC by USB cable.

#### Step3: Image Capturing
To turn on RICOH THETA S as the live streaming mode, please hold the bottom buttom at side for about 5 senconds and push the top buttom.(Detail is shown in the instrunction sheet of RICOH THETA S.)

To capture the image from RICOH THETA S, we used the open source in ROS, cv_camera_node(http://wiki.ros.org/cv_camera).
The subscribed topic name of the image is "/cv_camera_node1/image". We recommend that the flame rate of image is 3 fps.

#### Step4: Publishing subgoal image
DVMPC can follow the visual trajectory, which is consructed by the time consecutive 360-degree images.
So, before the navigation, you need to collect the visual trajectory by tele-operation of the robot.
Our code subscribes the visual trajectory as "/cv_camera_node2/image_ref".
Therefore, you need to feed the subgoal image from the visual trajectory into "/cv_camera_node2/image_ref".


#### Step5: Runing DVMPC
The last process to have the navigation is just to run our algorithm.

python dvmpc.py

The published topic name for the velocity reference is "/cmd_vel_mux/input/ref_GONetpp".
"img_ref" is the topic name for the current and subgoal images. 
And, front and back predicted images for 8 steps by VUNet-360 are published as "img_genf" and "img_genb".
If your implementation sounds correct, the 8-th predicted images should be similar to the subgoal image.

License
=================
The codes provided on this page are published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License(https://creativecommons.org/licenses/by-nc-sa/3.0/). This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license. If you are interested in commercial usage you can contact us for further options. 

Citation
=================

If you use DVMPC's software or database, please cite:

