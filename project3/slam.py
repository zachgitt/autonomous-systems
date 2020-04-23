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
import random
import pdb

# TODO: change hit/miss ratio to .8/.3

TEST_ROWS = 385 + 100

def frange(start, stop, step):
    i = start
    while i <= stop:
        yield i
    i += step

class Robot:

    # Deadreckoning Hit/Miss = 1.1/0.05

    # Map height and width are in centimeters
    def __init__(self,
        encoder_in,
        lidar_in,
        imu_in,
        height=8001,
        width=8001,
        rmax=20,
        rmin=-20,
        alpha_hit=1.1,
        alpha_miss=0.05,
        init_hit=5,
        init_miss=-1,
        hit_thresh=1,
        sigma=1,
        sigma_th=3*pi/180,
        num_particles=40,
        discretize=2):

        # TODO: change miss back to 0.05, increasing to .3 should decrease whiteness of map though
        # Parameter options:
        # rmax/rmin: increase range so false positive black can become white
        # alpha_hit/alpha_miss: increase ratio so white doesn't overpower
        # init_hit/init_miss: increase to strengthen first lidar
        # hit_thresh: increase and it becomes harder to paint new points
        # sigma: decrease to consolidate points

        # Save map parameters
        self.height = height
        self.width = width
        self.rmax = rmax
        self.rmin = rmin
        self.alpha_hit = alpha_hit
        self.alpha_miss = alpha_miss
        self.slam_map = None
        self.column_names = ['dist' + str(j) for j in range(1081)]
        self.angles = []
        self.init_hit = init_hit
        self.init_miss = init_miss
        self.hit_thresh = hit_thresh
        self.sigma = sigma
        self.sigma_th = sigma_th
        self.num_particles = num_particles
        self.discretize = discretize

        # Save angles, every quarter degree, sweeps 270 degrees, store radians
        i = -135
        while i <= 135:
            self.angles.append(i * pi/180)
            i += 0.25

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

    def calculate_deadreckon_pose(self):
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

        # Center wheel to center wheel width (mm)
        width = 725  # Width of robot (#760 #750 #740 #720 #700 #900 #800 #733 #393.7)
        diam = 254  # Wheel diameter
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
            x_glob += self.lidar['dx'].iloc[i] / 10  # Convert mm to cm
            y_glob += self.lidar['dy'].iloc[i] / 10  # Convert mm to cm

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
        msg = 'Increase the map size! i={}, j={}, height={}, width={}'.format(i, j, height, width)
        assert i >= 0 and j >= 0 and i < height and j < width, msg

        return int(i), int(j)

    def get_column_names(self):
        return self.column_names

    def get_angles(self):
        return self.angles

    def calculate_deadreckon_map(self):
        """
        Height and width in centimeters.
        """
        # Construct map
        self.deadreckon_map = np.zeros(shape=(self.height, self.width)) # must be odd indexed

        # Initialize vars
        cols = self.get_column_names()
        angles = self.get_angles()

        # Iterate rows
        for row in range(self.lidar.shape[0]):

            print('step:' + str(row) + '/' + str(self.lidar.shape[0]))

            # Save globals for this row
            glob_theta = self.lidar['theta'].iloc[row]
            glob_x = self.lidar['x'].iloc[row]
            glob_y = self.lidar['y'].iloc[row]
            glob_i, glob_j = self.map_indices(glob_x, glob_y, self.height, self.width)

            # Skip when there is no movement
            if (row > 0):
                if self.lidar['x'].iloc[row] == self.lidar['x'].iloc[row-1] and \
                   self.lidar['y'].iloc[row] == self.lidar['y'].iloc[row-1]:
                    continue

            # Iterate all angles
            for col, angle in zip(cols, angles):
                dist = self.lidar[col].iloc[row] * 100 # Convert m to cm
                x = dist * cos(glob_theta + angle) + glob_x
                y = dist * sin(glob_theta + angle) + glob_y

                # Add hit
                i, j = self.map_indices(x, y, self.height, self.width)
                if self.deadreckon_map[i][j] + self.alpha_hit < self.rmax:
                    self.deadreckon_map[i][j] += self.alpha_hit

                # Add misses
                misses = list(bresenham(i, j, glob_i, glob_j))
                for miss in misses[1:]: # skip the end
                    if self.deadreckon_map[miss[0]][miss[1]] - self.alpha_miss > self.rmin:
                        self.deadreckon_map[miss[0]][miss[1]] -= self.alpha_miss

    def print_map(self, map, idx, map_folder, name, positions=None):
        # Map range (rmin, rmax) to (white, black) aka (miss, hit)
        img = np.round(np.interp(map, [self.rmin, self.rmax], [255, 0]))
        img = Image.fromarray(img.astype('uint8'))

        # Convert to rgb
        rgbimg = Image.new("RGBA", img.size)
        rgbimg.paste(img)

        # Save pose as red dots
        if name == 'deadreckon':
            for row in range(self.lidar.shape[0]):
                i, j = self.map_indices(self.lidar['x'].iloc[row], self.lidar['y'].iloc[row], self.height, self.width)
                rgbimg.putpixel((j, i), (255, 0, 0))
        elif name == 'slam':
            for position in positions:
                rgbimg.putpixel((position[1], position[0]), (255, 0, 0))

        rgbimg.putpixel(self.map_indices(0,0, self.height, self.width), (0,0,255))
        rgbimg.save(map_folder + name + str(idx) + '.png')


    def print_particle_positions(self, idx, best_positions):

        # Live update map range (rmin, rmax) to (white, black) aka (miss, hit)
        img = np.round(np.interp(self.slam_map, [self.rmin, self.rmax], [255, 0]))
        img = Image.fromarray(img.astype('uint8'))

        # Convert to rgb
        rgbimg = Image.new("RGBA", img.size)
        rgbimg.paste(img)

        # Save best position as purple
        for position in best_positions:
            rgbimg.putpixel((position[1], position[0]), (128, 0, 128))

        for t in range(TEST_ROWS):
            # Save best position as purple


            for p in range(self.num_particles):
                name = 'particle' + str(p)
                name_x = name + '_x'
                name_y = name + '_y'

                random.seed(p)

                i, j = self.map_indices(self.lidar[name_x].iloc[t], self.lidar[name_y].iloc[t], self.height, self.width)
                rgbimg.putpixel((j, i), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

                # if p % 3 == 0:
                #     rgbimg.putpixel((j, i), (255, 0, 0))
                # elif p % 3 == 1:
                #     rgbimg.putpixel((j, i), (0, 255, 0))
                # elif p % 3 == 2:
                #     rgbimg.putpixel((j, i), (0, 0, 255))
        rgbimg.save(os.getenv('HOME') + '/Desktop/slam' + str(idx) + '.png')
        print('LIVE MAP PRINTED')

    def discretizer(self, i, j):
        size = self.discretize
        adj = [] # TODO: return list of tuples, adjacent i/j pairs

        # TODO: find offset of block this belongs to: i.e. block_row=20, block_col=40
        #  width and height should be a multiple of block_row and block_col

        # TODO: record all adjacent cells in the block

        return adj

    def init_map(self, init_hit, init_miss):

        # Initialize slam map
        self.slam_map = np.zeros(shape=(self.height, self.width))
        glob_i, glob_j = self.map_indices(0, 0, self.height, self.width)
        for col, angle in zip(self.get_column_names(), self.get_angles()):
            dist = self.lidar[col].iloc[0] * 100 # Convert meter to centimeter
            x = dist * cos(angle)
            y = dist * sin(angle)

            # Add initial hit
            i, j = self.map_indices(x, y, self.height, self.width)
            for cell in self.discretizer(i, j):
                self.slam_map[cell[0]][cell[1]] = init_hit

            # for d1 in range(-self.discretize, self.discretize):
            #     for d2 in range(-self.discretize, self.discretize):

            # Add initial misses
            misses = list(bresenham(i, j, glob_i, glob_j))
            for miss in misses[1:]:  # skip the end
                self.slam_map[miss[0]][miss[1]] = init_miss

    # Initialize n particles with noise
    def init_particles(self, N):

        # Add x, y, theta column for each particle
        for i in range(N):
            name = 'particle' + str(i)
            name_x = name + '_x'
            name_y = name + '_y'
            name_th = name + '_th'
            self.lidar[name_x] = float('NaN')
            self.lidar[name_y] = float('NaN')
            self.lidar[name_th] = float('NaN')

            # Initialize pose of each particle
            self.lidar[name_x].iloc[0] = 0
            self.lidar[name_y].iloc[0] = 0
            self.lidar[name_th].iloc[0] = 0

    def update_particles(self, t, N, sigma, hit_thresh):

        # Calculate movement
        width = 725
        diam = 254
        eL = pi * diam * (self.lidar['LeftTicks'].iloc[t] / 360)
        eR = pi * diam * (self.lidar['RightTicks'].iloc[t] / 360)
        th = (eR - eL) / width

        # Update each particle pose
        positions = [] # TODO: remove
        counts = np.zeros(N)
        for n in range(N):
            name_x = 'particle' + str(n) + '_x'
            name_y = 'particle' + str(n) + '_y'
            name_th = 'particle' + str(n) + '_th'

            # No movement
            if th == 0:
                self.lidar[name_th].iloc[t] = self.lidar[name_th].iloc[t-1]
                self.lidar[name_x].iloc[t] = self.lidar[name_x].iloc[t-1]
                self.lidar[name_y].iloc[t] = self.lidar[name_y].iloc[t-1]
                positions.append((self.lidar[name_x].iloc[t], self.lidar[name_y].iloc[t])) # TODO: remove
                continue

            # Add movement to particle (convert mm to cm)
            particle_th = self.lidar[name_th].iloc[t-1] + th + np.random.normal(scale=self.sigma_th)
            particle_x = self.lidar[name_x].iloc[t-1] + np.cos(particle_th) * (1 / 10) * (eL + eR) / 2 + np.random.normal(scale=sigma)
            particle_y = self.lidar[name_y].iloc[t-1] + np.sin(particle_th) * (1 / 10) * (eL + eR) / 2 + np.random.normal(scale=sigma)

            # Save updated particle
            self.lidar[name_th].iloc[t] = particle_th
            self.lidar[name_x].iloc[t] = particle_x
            self.lidar[name_y].iloc[t] = particle_y
            positions.append((self.lidar[name_x].iloc[t], self.lidar[name_y].iloc[t])) # TODO: remove

            # Count particle hits
            for col, angle in zip(self.get_column_names(), self.get_angles()):

                # Determine hit location
                dist = self.lidar[col].iloc[t] * 100 # Convert meter to centimeter
                x = dist * cos(particle_th + angle) + particle_x
                y = dist * sin(particle_th + angle) + particle_y
                i, j = self.map_indices(x, y, self.height, self.width)

                # Check this hit aligns well with map hits
                if self.slam_map[i][j] >= hit_thresh: # TODO: how do you adjust this threshold? It will only match with the map if it aligns with t=0 hit
                    # Save count
                    counts[n] += 1


        # # TODO: remove this live update of map
        # if t > 380:
        #     # Live update map range (rmin, rmax) to (white, black) aka (miss, hit)
        #     img = np.round(np.interp(self.slam_map, [self.rmin, self.rmax], [255, 0]))
        #     img = Image.fromarray(img.astype('uint8'))
        #
        #     # Convert to rgb
        #     rgbimg = Image.new("RGBA", img.size)
        #     rgbimg.paste(img)
        #     for idx, position in enumerate(positions):
        #         i, j = self.map_indices(position[0], position[1], self.height, self.width)
        #         if idx % N == 0:
        #             rgbimg.putpixel((j, i), (255, 0, 0))
        #         elif idx % N == 1:
        #             rgbimg.putpixel((j, i), (0, 255, 0))
        #         elif idx % N == 2:
        #             rgbimg.putpixel((j, i), (0, 0, 255))
        #     rgbimg.save(os.getenv('HOME') + '/Desktop/livemap.png')
        #     # TODO: remove up to here

        # Check zero sum
        if not counts.any():
            return counts

        # Calculate weights
        weights = counts / counts.sum()
        return weights

    # Anything closer than nearby (centimeters) will not be added to the map
    def update_map(self, t, weights, nearby=20):

        # Find best particle
        idx = np.where(weights == np.amax(weights))[0][0]

        # Save particle vars
        name_x = 'particle' + str(idx) + '_x'
        name_y = 'particle' + str(idx) + '_y'
        name_th = 'particle' + str(idx) + '_th'
        particle_x = self.lidar[name_x].iloc[t]
        particle_y = self.lidar[name_y].iloc[t]
        particle_th = self.lidar[name_th].iloc[t]
        particle_i, particle_j = self.map_indices(particle_x, particle_y, self.height, self.width)

        print('Best Particle (x,y) = (' + str(particle_x) + ',' + str(particle_y) + ')')
        ratio = 0 # TODO: remove

        positions = []

        # Iterate all angles
        for col, angle in zip(self.get_column_names(), self.get_angles()):

            # Skip nearby hits  # TODO: nearby hits still seem to exist
            dist = self.lidar[col].iloc[t] * 100  # Convert meter to centimeter
            if dist < nearby:
                ratio += 1
                continue

            # Determine map indices
            x_hit = dist * cos(particle_th + angle) + particle_x
            y_hit = dist * sin(particle_th + angle) + particle_y
            i_hit, j_hit = self.map_indices(x_hit, y_hit, self.height, self.width)

            # Add hit
            if self.slam_map[i_hit][j_hit] + self.alpha_hit < self.rmax:
                for cell in self.discretizer(i_hit, j_hit):
                    self.slam_map[cell[0]][cell[1]] += self.alpha_hit

            # Add misses
            misses = list(bresenham(i_hit, j_hit, particle_i, particle_j))
            for miss in misses[1:]:  # skip the origin
                if self.slam_map[miss[0]][miss[1]] - self.alpha_miss > self.rmin:
                    self.slam_map[miss[0]][miss[1]] -= self.alpha_miss

        print('Nearby Ratio: ' + str(ratio/1081))# TODO: remove

        # Return position
        position = (particle_i, particle_j)
        return position

    # If number of effective particles is too low, resample
    def needs_resampling(self, weights, N, thresh=0.5):
        top = sum(weights) ** 2
        bottom = sum([weight ** 2 for weight in weights])
        ratio = top / bottom

        # Ratio is between 1-N
        return ratio < thresh * N

    # Resample the particles and add noise
    def resample(self, weights, N, t):

        # Determine boundaries
        bounds = []
        sum = 0
        for weight in weights:
            sum += weight
            bounds.append(sum)

        # Save resampled indices
        indices = []
        for i in range(N):
            val = sum * i / N
            idx = 0
            while val > bounds[idx]:
                idx += 1
            indices.append(idx)

        # Copy x, y, theta values of sample
        sample = []
        for idx in indices:
            name_x = 'particle' + str(idx) + '_x'
            name_y = 'particle' + str(idx) + '_y'
            name_th = 'particle' + str(idx) + '_th'
            x = self.lidar[name_x].iloc[t]
            y = self.lidar[name_y].iloc[t]
            th = self.lidar[name_th].iloc[t]
            sample.append((x, y, th))

        # Paste x, y, theta values of sample
        for i in range(N):
            name_x = 'particle' + str(i) + '_x'
            name_y = 'particle' + str(i) + '_y'
            name_th = 'particle' + str(i) + '_th'
            self.lidar[name_x].iloc[t] = sample[i][0]
            self.lidar[name_y].iloc[t] = sample[i][1]
            self.lidar[name_th].iloc[t] = sample[i][2]

    def slam(self, idx):
        """
        param N: Number of particles
        """
        N = self.num_particles

        # Initialize map
        self.init_map(self.init_hit, self.init_miss)

        # Initialize particles with noise
        self.init_particles(N)

        # Save positions
        positions = []

        # Run slam over each time step
        for t in range(1, self.lidar.shape[0]):
            print('Robot' + str(idx) + ' t=' + str(t) + '/' + str(self.lidar.shape[0]))

            # Skip timesteps with no movement
            #if (t > 0):
                #if self.lidar['x'].iloc[t] == self.lidar['x'].iloc[t - 1] and \
                   #self.lidar['y'].iloc[t] == self.lidar['y'].iloc[t - 1]:
                    #print('No Movement')
                    #continue

            if t >= TEST_ROWS: # TODO: remove
                break

            # Determine weights of each particle
            weights = self.update_particles(t, N, self.sigma, self.hit_thresh)
            if not weights.any():
                # Skip if no particles matched well
                print('No Movement')
                continue

            # Update the map with the best particle
            position = self.update_map(t, weights)
            positions.append(position)
            if self.needs_resampling(weights, N, thresh=0.5): # TODO: how to use this threshold?
                self.resample(weights, N, t)
                print('resampling')

        return positions

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
    slam = True
    train = True

    # Run train or test
    if train:
        input_folder = base + '/train/'
        output_folder = base + '/maps_train/'
    else:
        input_folder = base + '/test/'
        output_folder = base + '/maps_test/'


    # Read data
    robots = read_data(input_folder)

    # Calculate distances
    for i, robot in enumerate(robots):

        # Precalculation
        robot.accumulate_ticks()
        robot.calculate_deadreckon_pose()

        # Slam
        if slam:
            positions = robot.slam(i)
            robot.print_map(robot.slam_map, i, output_folder, 'slam', positions)
            robot.print_particle_positions(i, positions)
        else:
            # Dead reckoning
            robot.calculate_deadreckon_map()
            robot.print_map(robot.deadreckon_map, i, output_folder, 'deadreckon')


if __name__ == '__main__':
    main()
