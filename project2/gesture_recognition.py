from os import listdir
from os.path import isfile, join
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import numpy as np


class Gesture:

    # Define initializer
    def __init__(self, df_in, label_in, file_in):
        self.df = df_in
        self.label = label_in
        self.file = file_in
        self.codewords = []

    def get_df(self):
        return self.df

    def get_label(self):
        return self.label

    def get_file(self):
        return self.file

    def set_codewords(self, cluster_indices):
        assert(len(cluster_indices) == self.df.shape[0])
        self.df['codewords'] = cluster_indices


def file_to_jpg(file):
    return file.split('.')[0] + '.jpg'


# Removes .txt and
def get_label(file):

    # Manually determine label by first letter
    name = file.split('.')[0]
    letter = name[0]
    label = None

    # beat3/beat4
    if letter == 'b':
        label = name[:5]
    # circle
    elif letter == 'c':
        label = name[:6]
    # eight
    elif letter == 'e':
        label = name[:5]
    # inf(inity)
    elif letter == 'i':
        label = name[:3]
    # wave
    elif letter == 'w':
        label = name[:4]
    # Error
    else:
        assert 0

    return label


# Plots the accelerometer and gyroscope data against the time data
def plot_gestures(gestures, folder):

    # Skip plotting
    if listdir(folder):
        return

    for gesture in gestures:
        df = gesture.get_df()
        plot = sns.lineplot(x='ts', y='value', hue='variable', data=pd.melt(df, ['ts']))
        fig = plot.get_figure()
        fig.savefig(folder + file_to_jpg(gesture.get_file()))
        plt.clf()


"""
Returns imu files as pandas dataframes.
"""
def load_txt_files(folder):

    # Gestures are dataframes with labels
    gestures = []

    # Load directory files and sort
    files = sorted(listdir(folder))
    for file in files:
        path = join(folder, file)
        label = get_label(file)
        if isfile(path):
            df = pd.read_csv(path, sep='\t', names=['ts', 'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz'])
            g = Gesture(df, label, file)
            gestures.append(g)

    # Return imu dataframes
    return gestures


# Return the center for k cluster means
def kmeans(points, k):

    # Assign k centers to random points
    centers = points.sample(k)

    # Iterate until centers unchanged
    delta = 1
    epsilon = 0.001
    while delta > epsilon:
        # Assign points to closest center
        tree = cKDTree(centers)
        _, indices = tree.query(points)

        # Calculate cluster center as mean
        clusters = points.assign(indices=indices)
        new_centers = clusters.groupby(['indices']).mean()

        # Update delta to converge
        delta = np.linalg.norm(centers - new_centers.values)
        centers = new_centers

    return centers


# Save the codeword (cluster) each gesture point belongs to
def discretize(gestures, k):

    # Get cluster centers
    centers = kmeans(get_points(gestures), k)
    tree = cKDTree(centers)

    # Save codeword for each point (cluster)
    for gesture in gestures:
        _, indices = tree.query(gesture.get_df().drop(columns=['ts']))
        gesture.set_codewords(indices)

    return gestures


def get_points(gestures):
    # Concatenate all points
    points = pd.DataFrame()
    for gesture in gestures:
        points = pd.concat([points, gesture.get_df().drop(columns=['ts'])], ignore_index=True)
    return points


"""
Converts measurements into kmeans.
"""
def plot_kmeans(points, folder):

    # Skip plotting
    if listdir(folder):
        return

    # Init range [inclusive, exclusive)
    k_start = 1
    k_end = 101

    # Try different k values, tracking variance
    variances = []
    for k in range(k_start, k_end):

        # Track k centers that give lowest total variance
        #best_centers = [None]*k
        #min_var = math.inf

        # Assign k centers to random points
        centers = points.sample(k)

        # Iterate until centers unchanged
        delta = 1
        epsilon = 0.001
        variance = -1
        while delta > epsilon:
            # Assign points to closest center
            tree = cKDTree(centers)
            _, indices = tree.query(points)

            # Calculate cluster center as mean
            clusters = points.assign(indices=indices)
            new_centers = clusters.groupby(['indices']).mean()

            # Update delta to converge
            delta = np.linalg.norm(centers - new_centers.values)
            centers = new_centers

            # Calculate variance
            variance = clusters.groupby(['indices']).var().sum(axis=1).values[0]

        variances.append(variance)

        print("k=" + str(k) + " variance=" + str(variance))

    # Plot k vs. reduction in variance (elbow method to find optimal k)
    reductions = []
    for v in variances:
        reduction = variances[0] - v
        reductions.append(reduction)
    plt.plot(np.arange(k_start, k_end), reductions, '-o')
    plt.savefig(folder + 'kmeans.jpg')


def transition_matrix():
    return T


def emission_matrix():
    return E


def initial_probs():
    return Pi


def expectation_maximization(gestures, label):

    # Initialize model (lambda)
    model = None
    T = transition_matrix()
    E = emission_matrix()
    Pi = initial_probs()

    # Stop when likelihood plateus
    likelihoods = []
    likelihood_increases = True
    while likelihood_increases:

        # Expectation step: calculate parameters
        alpha = None
        beta = None
        gamma = None
        xi = None

        # Maximization step: improve T, E, Pi (lambda)

        # Save likelihoods
        likelihood = None
        likelihoods.append(likelihood)

    # Plot likelihood values

    return model


def main():

    # Configuration
    base = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project2'
    train_multiple_folder = base + '/train_multiple/'
    train_single_folder = base + '/train_single/'
    plot_folder = base + '/imu_plots/'
    kmeans_plot_folder = base + '/kmeans_plot/'
    optimal_k = 10

    # Load train imu data
    gestures = []
    gestures += load_txt_files(train_multiple_folder)
    gestures += load_txt_files(train_single_folder)

    # Plot gestures
    plot_gestures(gestures, plot_folder)

    # Discretize measurements with kmeans
    plot_kmeans(get_points(gestures), kmeans_plot_folder)
    gestures = discretize(gestures, optimal_k)

    # Model each type
    models = []
    for label in ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']:
        model = expectation_maximization(gestures, label)
        models.append(model)

    # Load test imu data

    # Discretize test data

    # Predict test data (most probable model)


if __name__ == '__main__':
    main()
