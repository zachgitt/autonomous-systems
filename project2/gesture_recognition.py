from os import listdir
from os.path import isfile, join
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
import numpy as np
import pickle


class Model:

    def __init__(self, A_in, B_in, Pi_in, label_in):
        self.A = A_in
        self.B = B_in
        self.Pi = Pi_in
        self.label = label_in

    def split_model(self):
        return self.A, self.B, self.Pi


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

    def get_codewords(self):
        return self.df['codewords']


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

    km = KMeans(n_clusters=k).fit(get_points(gestures)) # TODO: remove SCIKIT

    # Save codeword for each point (cluster)
    for gesture in gestures:
        _, indices = tree.query(gesture.get_df().drop(columns=['ts']))
        indices = km.predict(gesture.get_df().drop(columns=['ts'])) # TODO remove
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


def get_class_codewords(gestures, label):
    codewords = []
    for gesture in gestures:
        if gesture.get_label() == label:
            codewords.extend(gesture.get_codewords())

    return codewords


def get_alpha(codewords, A, B, Pi):

    # Initialization (t=0)
    T = len(codewords)
    N = len(Pi)
    alpha = np.ndarray(shape=(T, N))
    alpha[0] = Pi * B[codewords[0]]

    # Save scales
    scales = np.ndarray(shape=(T,))
    scales[0] = np.nan_to_num(1/np.sum(alpha[0]), nan=100000, posinf=100000, neginf=100000)
    alpha[0] *= scales[0]

    # Induction (t+1=t)
    for t in range(1, T):
        alpha[t] = np.dot(alpha[t-1], A) * B[codewords[t]]
        # Scale alpha
        scales[t] = np.nan_to_num(1 / np.sum(alpha[t]), nan=100000, posinf=100000, neginf=100000)
        alpha[t] *= scales[t]

    return alpha, scales


def get_beta(codewords, A, B, Pi, scales):

    # Initialize (T=end)
    T = len(codewords)
    N = len(Pi)
    beta = np.ndarray(shape=(T, N))
    beta[T-1] = scales[T-1]

    # Induction
    for t in reversed(range(T-1)):
        for i in range(N):
            sum = 0
            for j in range(N):
                sum += A[i][j] * B[codewords[t+1]][j] * beta[t+1][j]
            beta[t][i] = scales[t] * sum
        #beta[t] = scales[t] * np.dot(A, B[codewords[t+1]].T).T * beta[t+1]

    return beta


def get_gamma(alpha, beta):

    # Initialize gamma
    T, N = alpha.shape
    gamma = np.ndarray(shape=(T, N))
    gamma[T-1] = alpha[T-1]

    # Iterate
    for t in range(T-1):
        gamma[t] = (alpha[t] * beta[t]) / np.dot(alpha[t], beta[t])

    return gamma


def get_xi(A, B, alpha, beta, codewords):

    # Initialize xi
    T, N = alpha.shape
    xi = np.zeros(shape=(T, N, N))

    # Iterate
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                xi[t][i][j] = alpha[t][i] * A[i][j] * B[codewords[t+1]][j] * beta[t+1][j]

        # Normalize
        norm = np.sum(xi[t])
        xi[t] = xi[t] / norm

    return xi


def update_A(A, gamma, xi):

    # Initialize lengths
    N = A.shape[0]
    T = gamma.shape[0]

    for i in range(N):
        for j in range(N):
            top = 0
            bottom = 0
            for t in range(T):
                top += xi[t][i][j]
                bottom += gamma[t][i]
            A[i][j] = top / bottom

    return A


def update_B(B, gamma, codewords):

    # Initialize lengths
    M, N = B.shape
    T = gamma.shape[0]

    for k in range(M):
        for j in range(N):
            top = 0
            bottom = 0
            for t in range(T):
                if codewords[t] == k:
                    top += gamma[t][j]
                bottom += gamma[t][j]
            B[k][j] = top / bottom

    return B


def likelihood_increased(likelihoods, delta=0.001):
    if len(likelihoods) < 2:
        return True

    return delta < (likelihoods[-1] - likelihoods[-2])


def plot_likelihoods(likelihoods, label, folder):
    # Create plot
    plt.plot(likelihoods)
    plt.xlabel('Iteration')
    plt.ylabel('Likelihood of ' + label.capitalize())
    plt.savefig(folder + label + '_likelihoods.jpg')
    plt.clf()


def expectation_maximization(codewords, N, M, label, folder):

    # Print
    print('Computing ' + label.capitalize())

    # Initialize model (lambda)
    A = np.random.rand(N, N)
    B = np.random.rand(M, N)
    Pi = np.zeros(N)
    A /= A.sum(axis=1, keepdims=1)
    B /= B.sum(axis=0, keepdims=1)
    Pi[0] = 1

    # Stop when likelihood plateus
    likelihoods = []
    while True:

        # Expectation step: calculate parameters
        alpha, scales = get_alpha(codewords, A, B, Pi)
        beta = get_beta(codewords, A, B, Pi, scales)
        gamma = get_gamma(alpha, beta)
        xi = get_xi(A, B, alpha, beta, codewords)

        # Maximization step: improve A, B (lambda)
        A = update_A(A, gamma, xi)
        B = update_B(B, gamma, codewords)

        # Save likelihoods
        likelihood = -np.sum(np.log(scales))
        print(likelihood)
        likelihoods.append(likelihood)
        if not likelihood_increased(likelihoods, delta=1000):
            break

    # Plot likelihood values
    plot_likelihoods(likelihoods, label, folder)

    # Return model
    model = Model(A, B, Pi, label)
    return model


def predict(gesture, models):

    #import pdb; pdb.set_trace()

    # Predict most probable model
    predictions = np.zeros(shape=len(models))
    for i, model in enumerate(models):
        A, B, Pi = model.split_model()
        codewords = gesture.get_codewords()
        alpha, scales = get_alpha(codewords, A, B, Pi)
        likelihood = -np.sum(np.log(scales))
        predictions[i] = likelihood

    return predictions


def get_models(gestures, labels, N, K, plot_folder, model_folder):

    # Unpickle models
    if listdir(model_folder):
        with open(model_folder + 'models.pkl', 'rb') as f:
            models = pickle.load(f)
        return models

    # Calculate models
    models = []
    for label in labels:
        codewords = get_class_codewords(gestures, label)
        model = expectation_maximization(codewords, N, K, label, plot_folder)
        models.append(model)

    # Pickle models
    with open(model_folder + 'models.pkl', 'wb') as f:
        pickle.dump(models, f)

    return models


def print_predictions(predictions, labels):
    df = pd.DataFrame(data=predictions, columns=labels)
    df2 = df.copy().astype('int32')
    for i, row in enumerate(df):
        order = np.argsort(-df.iloc[i]).values.tolist()
        for j, idx in enumerate(order):
            df2.iloc[i][idx] = int(j)
    print(df)
    print(df2)


def main():

    # Configuration
    base = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project2'
    train_multiple_folder = base + '/train_multiple/'
    train_single_folder = base + '/train_single/'
    plot_folder = base + '/imu_plots/'
    kmeans_plot_folder = base + '/kmeans_plot/'
    test_folder = base + '/test/'
    likelihood_plots = base + '/likelihoods/'
    models_folder = base + '/models/'
    labels = ['beat3', 'beat4', 'circle', 'eight', 'inf', 'wave']
    optimal_k = 30
    num_hidden_states = 6

    # Load train imu data
    gestures = []
    gestures += load_txt_files(train_multiple_folder)
    #gestures += load_txt_files(train_single_folder)

    # Plot gestures
    plot_gestures(gestures, plot_folder)

    # Discretize measurements with kmeans
    plot_kmeans(get_points(gestures), kmeans_plot_folder)
    gestures = discretize(gestures, optimal_k)

    # Model each type
    models = get_models(gestures, labels, num_hidden_states, optimal_k, likelihood_plots, models_folder)

    # Load test imu data
    test_gestures = load_txt_files(test_folder) # TODO: change to test folder

    # Discretize test data
    test_gestures = discretize(test_gestures, optimal_k)

    # Predict test data (most probable model)
    predictions = np.ndarray(shape=(len(test_gestures), len(models)))
    for i, test in enumerate(test_gestures):
        predictions[i] = predict(test, models)

    # Print predictions
    print_predictions(predictions, labels)

if __name__ == '__main__':
    main()
