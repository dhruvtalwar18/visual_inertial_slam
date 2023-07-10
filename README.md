# Visual Inertial SLAM

<h1><b> Overview </b></h1>
In this project, we tackled the Simultaneous Localization and Mapping (SLAM) problem by implementing a powerful technique known as the Bayes Filter. Our focus was on a mobile robot operating in an environment initially unfamiliar to it. To address this challenge, we developed a Visual Inertial Extended Kalman Filter, which leverages the power of Gaussian distribution estimation. This filter enables us to accurately estimate the pose of the robot and the positions of landmarks in real-time. By collecting these landmark feature points, we construct a detailed map of the environment, providing crucial spatial information for the robot's autonomous navigation and exploration.

The figures presented below illustrate and compare the the trajectory obtained through dead reckoning and the improved trajectory obtained through the utilization of the Extended Kalman Filter. The enhanced trajectory incorporates the landmarks detected by the filter, providing more accurate and reliable position estimates for the robot's navigation and mapping tasks.


<table>
  <tr>
    <td align="center">
      <img src="https://github.com/dhruvtalwar18/visual_inertial_slam/blob/main/images/Test_1.png" title="Comparison on Dataset 3" style="width: 400px; height: 400px;">
      <br>
      <p align="center">Fig.1 Dead reckoning vs VI SLAM dataset 3</p>
    </td>
    <td align="center">
      <img src="https://github.com/dhruvtalwar18/visual_inertial_slam/blob/main/images/Test_2.png" title="Comparison on Dataset 10" style="width: 400px; height: 400px;">
      <br>
      <p align="center">Fig.2 Dead reckoning vs VI SLAM dataset 10</p>
    </td>
  </tr>
</table>

<h1><b> Code Setup </b></h1>

Create a conda environment 
```
conda create -name visual_inertial_slam
conda activate visual_iinertial_slam
git clone https://github.com/dhruvtalwar18/visual_inertial_slam
cd visual_inertial_slam
pip install -r requirements.txt

```

Scripts implemented 

In the Trajectory_only folder run the trajectory.py as follows:
```
python3 trajectory.py 

```
The purpose of this script is just to form a path by the motion model implemented on the IMU data points and form a path.
<br>
In the Mapping_trajectory folder run the Mapping_trajectory.py script as follows

```
python3 Mapping_trajectory.py

```
The purpose of this script is to execute the IMU EKF predict step and the landmark update step for the specific problems outlined in the Project Guidelines document. Its main function is to perform these steps and provide the necessary output. To utilize the script, follow the usage instructions provided in the Project Guidelines.

<br>

In the Visual_SLAM folder run the visual_slam.py script as follows:
```
python3 visual_slam.py

```


The main objective of this script is to execute the complete Visual Inertial SLAM algorithm on the provided datasets. It achieves this by simultaneously performing the update step for both the IMU pose and the landmarks. To utilize the script and apply the algorithm to the datasets, please follow the provided usage instructions.



