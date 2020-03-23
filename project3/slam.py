import sys, os
sys.path.append(os.path.join(sys.path[0], 'ECE5242Proj3-train'))
from load_data import get_lidar, get_encoder, get_imu
from p3_util import replay_lidar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
from bresenham import bresenham
from PIL import Image
import pdb


class Robot:

    def __init__(self, encoder_in, lidar_in, imu_in):

        # Reformat encoder data
        self.encoder = pd.DataFrame(encoder_in).T
        self.encoder.rename(columns={
            0:'t',
            1:'ForwardRight',
            2:'ForwardLeft',
            3:'BackRight',
            4:'BackLeft'
        }, inplace=True)

        # Reformat lidar data
        height, width = len(lidar_in), len(lidar_in[0]['scan'])
        names = ['t'] + ['dist' + str(i) for i in range(width)]
        self.lidar = pd.DataFrame(np.zeros((height, width+1)), columns=names)
        for i in range(height):
            # Prepend time to distance scan data
            self.lidar.iloc[i] = np.insert(lidar_in[i]['scan'], 0, lidar_in[i]['t'])

        # Reformat imu data
        self.imu = pd.DataFrame(imu_in).T
        self.imu.rename(columns={
            0:'t',
            1:'Ax',
            2:'Ay',
            3:'Az',
            4:'Gx',
            5:'Gy',
            6:'Gz'
        }, inplace=True)

    def accumulate_ticks(self):
        """
        Between each lidar timestep, all encoder ticks are accumulated.
        """

        # Add new columns
        self.lidar['LeftTicks'] = float('NaN')
        self.lidar['RightTicks'] = float('Nan')

        # Initialize variables
        num_encoders, num_lidar = self.encoder.shape[0], self.lidar.shape[0]
        t0 = self.lidar['t'].iloc[0]
        i, j = 0, 0

        # Skip all pre-lidar encoder movements
        while self.encoder['t'].iloc[j] < t0:
            j += 1

        # Calculate leftTicks and rightTicks
        sumL, sumR = 0, 0
        while j < num_encoders and i < num_lidar:

            # Accumulate encoder movement
            if i+1 < num_lidar and self.encoder['t'].iloc[j] < self.lidar['t'].iloc[i+1]:
                sumL += (self.encoder['ForwardLeft'].iloc[j] + self.encoder['BackLeft'].iloc[j])/2
                sumR += (self.encoder['ForwardRight'].iloc[j] + self.encoder['BackRight'].iloc[j])/2
                j += 1

            # Save and reset movement
            else:
                self.lidar['LeftTicks'].iloc[i] = sumL
                self.lidar['RightTicks'].iloc[i] = sumR
                sumL, sumR = 0, 0
                i += 1

        # Save leftover ticks
        if sumL != 0 or sumR != 0:
            self.lidar['LeftTicks'].iloc[i] = sumL
            self.lidar['RightTicks'].iloc[i] = sumR

    def calculate_pose(self):
        """
        Using LeftTicks and RightTicks per lidar timestep,
        calculate change in theta, x, and y.
        """
        # Add dtheta, dx, dy column
        self.lidar['dtheta'] = float('NaN')
        self.lidar['dx'] = float('NaN')
        self.lidar['dy'] = float('NaN')

        # Add globals theta, x, y
        self.lidar['theta'] = float('NaN')
        self.lidar['x'] = float('NaN')
        self.lidar['y'] = float('NaN')

        # Center wheel to center wheel width
        diam = 254
        width = 393.7
        th_glob = 0
        x_glob = 0
        y_glob = 0
        for i in range(self.lidar.shape[0]):
            eL = pi * diam * (self.lidar['LeftTicks'].iloc[i] / 360)
            eR = pi * diam * (self.lidar['RightTicks'].iloc[i] / 360)
            th = (eR - eL) / width
            th_glob += th

            # Calculate local measurements
            self.lidar['dtheta'].iloc[i] = th
            self.lidar['dx'].iloc[i] = np.cos(th_glob) * (eL + eR) / 2
            self.lidar['dy'].iloc[i] = np.sin(th_glob) * (eL + eR) / 2
            x_glob += self.lidar['dx'].iloc[i] / 10
            y_glob += self.lidar['dy'].iloc[i] / 10

            # Calculate global measurements
            self.lidar['theta'].iloc[i] = th_glob
            self.lidar['x'].iloc[i] = x_glob
            self.lidar['y'].iloc[i] = y_glob

    def print_pose(self, i, map_folder):
        plt.plot(self.lidar['x'], self.lidar['y'])
        plt.savefig(map_folder + 'pose' + str(i) + '.png')
        plt.clf()

    def map_indices(self, x, y, height, width):
        """
        Center of map is xy = (0,0).
        Map is odd indexed.
        """
        i = -y + int(height/2)
        j = x + int(width/2)
        # Check out of bounds
        if i < 0 or j < 0 or i >= height or j >= width:
            assert(0, 'Increase the map size!')
        return int(i), int(j)

    def calculate_map(self, height=7501, width=7501, rmax=10, rmin=-10, alpha_hit=0.7, alpha_miss=0.3):
        """
        Height and width in centimeters.
        """
        # Construct map
        self.map = np.zeros(shape=(height, width)) # must be odd indexed

        # Initialize vars
        cols = ['dist' + str(j) for j in range(1081)]
        angles = [a*pi/180 for a in range(-135, 136)]

        # Iterate rows
        for row in range(self.lidar.shape[0]):

            print('pose:' + str(row) + '/' + str(self.lidar.shape[0]))

            # Save globals for this row
            glob_theta = self.lidar['theta'].iloc[row]
            glob_x = self.lidar['x'].iloc[row]
            glob_y = self.lidar['y'].iloc[row]
            glob_i, glob_j = self.map_indices(glob_x, glob_y, height, width)

            # Iterate all angles
            for col, angle in zip(cols, angles):
                dist = self.lidar[col].iloc[row] * 100 # Convert to centimeter
                x = dist * cos(glob_theta + angle) + glob_x
                y = dist * sin(glob_theta + angle) + glob_y

                # Add hit
                i, j = self.map_indices(x, y, height, width)
                if self.map[i][j] + alpha_hit < rmax:
                    self.map[i][j] += alpha_hit

                # Add misses
                misses = list(bresenham(i, j, glob_i, glob_j))
                for miss in misses[1:]: # skip the end
                    if self.map[miss[0]][miss[1]] - alpha_miss > rmin:
                        self.map[miss[0]][miss[1]] -= alpha_miss

    def print_map(self, idx, map_folder):
        # Map range (rmin, rmax) to (white, black) aka (miss, hit)
        img = np.round(np.interp(self.map, [-10, 10], [255, 0]))
        img = Image.fromarray(img.astype('uint8'))
        img.save(map_folder + 'map' + str(idx) + '.png')


def read_data(folder):

    # Determine number of files
    files = sorted(os.listdir(folder))
    num = int(len(files)/3)

    # Read each trial run
    robots = []
    for i in range(num):
        encoder = get_encoder(folder + files[i])
        lidar = get_lidar(folder + files[i + num])
        imu = get_imu(folder + files[i + 2*num])
        r = Robot(encoder, lidar, imu)
        robots.append(r)

    return robots


def main():
    # Configuration
    base = os.getcwd() + '/ECE5242Proj3-train'
    train_folder = base + '/train/'
    test_folder = base + '/test/'
    map_folder = base + '/maps/'

    # Read training
    robots = read_data(train_folder)

    # Calculate distances
    for i, robot in enumerate(robots):
        robot.accumulate_ticks()
        robot.calculate_pose()
        robot.print_pose(i, map_folder)
        robot.calculate_map()
        robot.print_map(i, map_folder)

if __name__ == '__main__':
    main()
