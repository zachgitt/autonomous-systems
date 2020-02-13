from os import listdir
from os.path import isfile, join
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


class Gesture:

    # Define initializer
    def __init__(self, df_in, label_in, file_in):
        self.df = df_in
        self.label = label_in
        self.file = file_in

    def get_df(self):
        return self.df

    def get_label(self):
        return self.label

    def get_file(self):
        return self.file


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
    for gesture in gestures:
        df = gesture.get_df()
        plot = sns.lineplot(x='ts', y='value', hue='variable', data=pd.melt(df, ['ts']))
        fig = plot.get_figure()
        fig.savefig(folder + file_to_jpg(gesture.get_file()))
        plt.show()


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


def main():

    # Configuration
    base = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project2'
    train_multiple_folder = base + '/train_multiple/'
    train_single_folder = base + '/train_single/'
    plot_folder = base + '/imu_plots/'

    # Load imu data
    gestures = []
    gestures += load_txt_files(train_multiple_folder)
    gestures += load_txt_files(train_single_folder)

    # Plot gestures
    plot_gestures(gestures, plot_folder)


if __name__ == '__main__':
    main()
