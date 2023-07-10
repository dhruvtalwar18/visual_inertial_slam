# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:22:19 2023

@author: dhruv
"""

import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

def load_data(file_name):
    '''
    function to read visual features, IMU measurements, and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic transformation from (left) camera to imu frame, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of the visual features
        linear_velocity = data["linear_velocity"] # linear velocity in body-frame coordinates
        angular_velocity = data["angular_velocity"] # angular velocity in body-frame coordinates
        K = data["K"] # intrinsic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # transformation from left camera frame to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam



def visualize_trajectory_2d1(pose,M,path_name="Unknown",show_ori=False):
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(10,10))
  n_pose = pose.shape[2]
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(M[0,:],M[1,:],marker='o',label="landmarks")
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,int(n_pose/50)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax


import numpy as np

def d_x_matrix(x):
    """
    Calculates the 4x4 matrix d_q given a 4D vector q.
    
    Args:
    - q: 4D numpy array representing a homogeneous point in camera coordinates.
    
    Returns:
    - d_q: 4x4 numpy array representing the derivative matrix of q.
    """
    d_x = (1/x[2]) * np.array([[1, 0, -x[0]/x[2], 0],
                                [0, 1, -x[1]/x[2], 0],
                                [0, 0, 0, 0],
                                [0, 0, -x[3]/x[2], 1]])
    return d_x

def hat_map(x):
    x_hat = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
    return x_hat





import numpy as np

def initialize_values(filename):
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam = load_data(filename)
    features = features[:, 0:features.shape[1]:50, :]
    odom = np.zeros((4, 4, t.size - 1))
    odom_mu = np.identity(4)
    odom_sigma = np.identity(6)

    _, num, _ = features.shape  
    matrix = np.zeros((3, 4))
    matrix[:3, :3] = np.identity(3)
    projection_P = matrix

    z_default = np.array([-1, -1, -1, -1]) 

    map_mu_index = 0

    K_4x3 = np.vstack((K[0], K[1], K[0], K[1]))
    neg_fsu_b = -331.5325566
    camera_matrix = np.column_stack((K_4x3, np.array([[0], [0], [neg_fsu_b], [0]])))
    

    map_sigma = np.tile(100 * np.identity(3), (num, 1, 1))
    map_mu = np.zeros((4, num))

    z_tilt = np.full((4, num), -1)
    z_actual = np.full((4, num), -1)
    z_predicted = np.full((4, num), -1)
    V = 10*np.eye(4)

    return t, features, linear_velocity, angular_velocity, K, b, imu_T_cam, odom, odom_mu, odom_sigma, \
           num, projection_P, z_default, map_mu_index, camera_matrix, map_sigma, map_mu, z_tilt, \
           z_actual, z_predicted, V




import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv,pinv
from scipy.linalg import solve
if __name__ == '__main__':
    filename = "/content/drive/MyDrive/PR3/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam, odom, odom_mu, odom_sigma, \
    num, projection_P, z_default, map_mu_index, camera_matrix, map_sigma, map_mu, z_tilt, \
    z_actual, z_predicted, V = initialize_values(filename)

    H_SLAM = np.array([[0]*(3*num+6) for _ in range(4*num)])
    W_noise = 2* np.identity(6)
    for i in range(t.size-1):
      tau = t[0,i+1]-t[0,i]
    # hat maps for angular velocities
      ang_vel_hat = hat_map(angular_velocity[:, i])
      lin_vel_hat = hat_map(linear_velocity[:,i])
      wv =  np.hstack((ang_vel_hat,lin_vel_hat))
      zeros_matrix = np.zeros((3, 3))
      ow = np.hstack((zeros_matrix,ang_vel_hat))
      u_curly = np.vstack((wv,ow))
    # Stack the hat maps with linear velocity and zeros to get the 4x4 matrix
      u_hat = np.vstack((np.hstack((ang_vel_hat, linear_velocity[:, i, np.newaxis])), np.zeros((1, 4))))
      R = expm(-tau * u_hat)
      odom_mu = R @ odom_mu
      odom[:, :, i] = np.linalg.inv(odom_mu)    
      odom_sigma = expm(-tau*u_curly) @ odom_sigma @ expm(-tau*u_curly).T + W_noise 

    #ang_vel_hat = hat_map(angular_velocity[:, i])
    # Stack the hat maps with linear velocity and zeros to get the 4x4 matrix
    #u_hat = np.vstack((np.hstack((ang_vel_hat, linear_velocity[:, i].reshape((-1,1)))),np.zeros((1, 4))))
    #R = expm(-tau* u_hat)
    #mu = R @ mu
    #traj[:, :, i] = inv(mu)
      actual_z = features[:,:,i] 
      H = np.zeros((4,3))
    #o_T_l = imu_T_cam @ odom_mu

    
      for landmark_idx in range(num):
        idx = np.where(actual_z[:,landmark_idx] != z_default)[0]
        if len(idx) > 0 and np.all(z_tilt[idx,landmark_idx] == z_default):
          z_tilt[idx,landmark_idx] = actual_z[idx,landmark_idx]
          homogeneous_pixel_coords = np.linalg.pinv(camera_matrix) @ z_tilt[:,landmark_idx]
          depth_remove = homogeneous_pixel_coords[3]
          homogeneous_pixel_coords = (1/depth_remove)*homogeneous_pixel_coords
          map_mu[:,map_mu_index] = odom[:, :, i] @ imu_T_cam @ homogeneous_pixel_coords
        #a = inv(imu_T_cam) @ odom_mu @ z_tilt[idx,landmark_idx]
        #a = a/a[3]
        #map_mu[:,map_mu_index]   = camera_matrix @ a
          map_mu_index  += 1

        else:
          idx = np.where(actual_z[:,landmark_idx] != z_default)[0]
          if len(idx) > 0 and np.all(z_tilt[idx,landmark_idx] == z_default):
            pi_matrix = inv(imu_T_cam) @ inv(odom[:, :, i]) @ map_mu[:, landmark_idx]   #
            z_predicted[:, landmark_idx] = camera_matrix @ (pi_matrix / pi_matrix[2])          
            dpi_dq = d_x_matrix(pi_matrix)
          #print(dpi_dq.shape)
            H = camera_matrix @ dpi_dq @ inv(imu_T_cam) @ inv(odom[:, :, i]) @ map_mu[:, landmark_idx] @  inv(imu_T_cam) @ inv(odom[:, :, i])@ projection_P.T
            K = map_sigma[landmark_idx] @ H.T @ solve(H @ map_sigma[landmark_idx] @ H.T + V, H @ map_sigma[landmark_idx])
            map_mu[:, landmark_idx] = map_mu[:, landmark_idx] + K @ (actual_z[:, landmark_idx] - z_predicted[:, landmark_idx])
            map_sigma[landmark_idx] = (np.identity(3)-K@H)@map_sigma[landmark_idx]

          #print(K.shape)
          #print(H.shape)
          #print(imu_T_cam.shape)
          #print(z_predicted.shape)
          #print(.shape)
          #print(dpi_dq.shape)

          
      


visualize_trajectory_2d1(odom,map_mu,path_name="Dataset_03")