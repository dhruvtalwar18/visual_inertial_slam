<img width="265" alt="image" src="https://github.com/dhruvtalwar18/visual_inertial_slam/assets/85046542/75306a33-d6ec-49cb-bb60-60e59bc5bb3b"># Visual Inertial SLAM

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
