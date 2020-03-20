# Jinwook Huh

import load_data as ld
import p3_util as ut
imu_ts, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z	 = ld.get_imu('train/imu20.mat')

enc_ts, FR, FL, RR, RL = ld.get_encoder('train/Encoders20.mat')

lidar = ld.get_lidar('train/Hokuyo20.mat')
ut.replay_lidar(lidar)
