import numpy as np
from scipy.linalg import expm
from numpy.linalg import inv,pinv
from scipy.linalg import solve
from pr3_utils import *
if __name__ == '__main__':
    filename = "/content/drive/MyDrive/PR3/10.npz"
    t, features, linear_velocity, angular_velocity, K, b, imu_T_cam, odom, odom_mu,\
    num, projection_P, z_default, map_mu_index, camera_matrix, map_sigma, map_mu, z_tilt, \
    z_actual, z_predicted, V = initialize_values(filename)

    W_noise = 2* np.identity(6)
    odom_sigma = np.identity(3*num+6)
    V = 10*np.identity(4*num)
    D = np.kron(np.eye(num), np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]]))

    for i in range(t.size-1):
      tau = t[0,i+1]-t[0,i]
      # hat maps for angular velocities
      ang_vel_hat = hat_map(angular_velocity[:, i])
      lin_vel_hat = hat_map(linear_velocity[:,i])
      wv =  np.hstack((ang_vel_hat,lin_vel_hat))
      zeros_matrix = np.zeros((3, 3))
      ow = np.hstack((zeros_matrix,ang_vel_hat))

    # Stack the hat maps with linear velocity and zeros to get the 4x4 matrix
      u_hat = np.vstack((np.hstack((ang_vel_hat, linear_velocity[:, i, np.newaxis])), np.zeros((1, 4))))
      u_curly = np.vstack((np.hstack((ang_vel_hat,lin_vel_hat)),np.hstack((np.zeros((3,3)),ang_vel_hat))))
      odom_sigma[3*num:3*num+6,3*num:3*num+6] = expm(-tau*u_curly)@odom_sigma[3*num:3*num+6,3*num:3*num+6]@(expm(-tau*u_curly)).T+W_noise



    #ang_vel_hat = hat_map(angular_velocity[:, i])
    # Stack the hat maps with linear velocity and zeros to get the 4x4 matrix
    #u_hat = np.vstack((np.hstack((ang_vel_hat, linear_velocity[:, i].reshape((-1,1)))),np.zeros((1, 4))))
    #R = expm(-tau* u_hat)
    #mu = R @ mu
    #traj[:, :, i] = inv(mu)
      z = features[:,:,i+1]
      H1 = np.zeros((4,3))
      H = np.zeros((4*num,3*num+6))
    #o_T_l = imu_T_cam @ odom_mu

    
      for landmark_idx in range(num):
        idx = np.where(z[:,landmark_idx] != z_default)[0]
        if len(idx) > 0 and np.all(z_tilt[idx,landmark_idx] == z_default):
          z_tilt[:,map_mu_index] = z[:,landmark_idx]
          homogeneous_pixel_coords = np.linalg.pinv(camera_matrix) @ z_tilt[:,landmark_idx]
          depth_remove = homogeneous_pixel_coords[3]
          homogeneous_pixel_coords = (1/depth_remove)*homogeneous_pixel_coords
          map_mu[:,map_mu_index] = inv(odom_mu) @ imu_T_cam @ homogeneous_pixel_coords
        #a = inv(imu_T_cam) @ odom_mu @ z_tilt[idx,landmark_idx]
        #a = a/a[3]
        #map_mu[:,map_mu_index]   = camera_matrix @ a
        #map_mu[:,landmark_idx] = landmark_world
          map_mu_index  += 1

        elif np.array_equal(idx, np.where(z[:,landmark_idx] != z_default)[0]):
          if len(idx) > 0 and np.all(z_tilt[idx,landmark_idx] == z_default):
            q = inv(imu_T_cam) @ odom_mu @ map_mu[:, landmark_idx] 
        
										 
        #z_predicted[:, landmark_idx] = camera_matrix @ (pi_matrix / pi_matrix[2])          
            dpi_dq = d_x_matrix(q)
          #print(dpi_dq.shape)
            H1 = camera_matrix @ dpi_dq @ inv(imu_T_cam) @ odom_mu @ D
          #K = map_sigma[landmark_idx] @ H.T @ solve(H @ map_sigma[landmark_idx] @ H.T + V, H @ map_sigma[landmark_idx])
          #map_mu[:, landmark_idx] = map_mu[:, landmark_idx] + K @ (actual_z[:, landmark_idx] - z_predicted[:, landmark_idx])
          #map_sigma[landmark_idx] = (np.identity(3)-K@H)@map_sigma[landmark_idx]

        else:
          H1 = np.zeros((4,3), dtype=np.float64)


        row_start = 4 * landmark_idx
        row_end = row_start + 4
        col_start = 3 * landmark_idx
        col_end = col_start + 3

        H[row_start:row_end, col_start:col_end] = H1

    
      s_qt = inv(imu_T_cam)@ odom_mu @ map_mu
      z_predicted = camera_matrix@(s_qt/s_qt[2,:])
      d = dot_fn(odom_mu @ map_mu)


      for k in range(num):
    # Check if the landmark measurement is the default value
        if np.all(map_mu[:,k] == z_default):
        # If so, set the corresponding block of slam_H to zero
            r_start, r_end = 4*k, 4*k+4
            c_start, c_end = 3*num, 3*num+6
            H[r_start:r_end, c_start:c_end] = np.zeros((4,6))
        else:
        # Calculate the necessary values
            landmark_position = map_mu[:,k]
            A = odom_mu @ landmark_position
            B = inv(imu_T_cam) @ A
            slam_d_q = d_x_matrix(B)
            dot_product = d[:,:,k]
        
        # Set the corresponding block of sH to the desired values
            H_start, H_end = 4*k, 4*k+4
            C_start, C_end = 3*num, 3*num+6
            H[H_start:H_end, C_start:C_end] = camera_matrix @ slam_d_q @ inv(imu_T_cam) @ dot_product

                                                                              

# Calculate the Kalman gain
    #Kalman_gain = odom_sigma @ H.T @ np.linalg.pinv(H @ odom_sigma @ H.T + V)
    Kalman_gain = compute_Kalman_gain(odom_sigma, H, V)

    case_test = np.any(z != -1, axis=0)

# Calculate the Kalman residual and update the estimates
    K_d = Kalman_gain @ ((((z - z_predicted) * case_test).swapaxes(0, 1)).reshape((4*num,1)))
    odom_mu = expm(hat1(K_d[num*3:num*3+6,:])) @ odom_mu
    map_mu = map_mu + (D @ K_d[0:3*num,:]).reshape((num,4)).T

# Update the covariance matrix
    odom_sigma = (np.identity(3*num+6) - Kalman_gain @ H) @ odom_sigma

    
    odom[:,:,i] = inv(odom_mu)
                          
visualize_trajectory_2d1(odom,map_mu)   
    
