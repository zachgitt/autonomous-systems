import cv2
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from roipoly import MultiRoi
import time
from mpl_toolkits.mplot3d import Axes3D
from math import exp, pi, sqrt
from numpy.linalg import det, inv
from numpy import dot
from skimage.measure import label, regionprops

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Time each function
start = time.time()


# Read train_images
def load_images(folder, color_space):

    # Convert train_images to color space
    color_code = None
    if color_space == 'hsv':
        color_code = cv2.COLOR_BGR2HSV
    else:
        color_code = cv2.COLOR_BGR2RGB

    # Read in train_images from folder
    images = []
    for filename in os.listdir(folder):

        # Skip . files
        if filename[0] == '.':
            continue

        # Default reads in as BGR (blue-green-red encoding)
        img_bgr = cv2.imread(os.path.join(folder, filename))
        if img_bgr is not None:
            # Convert to HSV
            img_hsv = cv2.cvtColor(img_bgr, color_code)
            images.append(img_hsv)

    print("Loading Images Complete " + str(time.time() - start))
    return images


def create_mask(image_folder, image_file):
    # Read image as grayscale
    img_bgr = cv2.imread(os.path.join(image_folder, image_file))
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Display UI
    fig = plt.figure()
    plt.imshow(img_gray, interpolation='nearest', cmap="Greys")
    plt.title("Click on the button to add a new ROI")

    # Draw multiple ROIs
    # NOTE: If you create more ROIs than the number of classes, only the
    #       last ROI will be saved
    multiroi_named = MultiRoi(roi_names=['red barrel-1', 'red barrel-2'])

    # Display ROIs
    masks = []
    plt.imshow(img_gray, interpolation='nearest', cmap="Greys")
    roi_names = []
    for name, roi in multiroi_named.rois.items():
        roi.display_roi()
        roi.display_mean(img_gray)
        roi_names.append(name)
        masks.append(roi.get_mask(img_gray))
    plt.legend(roi_names, bbox_to_anchor=(1.2, 1.05))
    plt.show()

    # Union masks (red barrel-1 and red barrel-2)
    mask = np.logical_or(masks[0], masks[1])
    return mask


def get_image_names(image_folder):
    names = []
    for image_file in os.listdir(image_folder):
        # Skip dot file
        if image_file[0] == '.':
            continue
        names.append(image_file[:-4])
    return names


# Read masks
def load_masks(image_folder, mask_folder):

    # Load masks for each image
    masks = []
    for image_file in os.listdir(image_folder):

        # Skip non png
        if image_file[-3:] != 'png':
            continue

        # Create mask if it does not exist
        mask_file = mask_folder + image_file[:-4] + '.npy'
        if not os.path.exists(mask_file):
            mask = create_mask(image_folder, image_file)
            np.save(mask_file, mask)

        # Read mask
        mask = np.load(mask_file, allow_pickle=True)
        masks.append(mask)

    print("Loading Masks Complete " + str(time.time() - start))
    return masks


# Print image[s] (useless)
def print_images(images):
    for image in images:
        plt.imshow(image)
        plt.show()


# For the provided train_images, highlight the masks in red (hsv space)
def print_masks(images, masks):
    for image, mask in zip(images, masks):
        image_copy = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i][j]:
                    # HSV (0, 255, 255)
                    image_copy[i][j][0] = 0
                    image_copy[i][j][1] = 255
                    image_copy[i][j][2] = 255
                else:
                    image_copy[i][j][0] = 0
                    image_copy[i][j][1] = 0
                    image_copy[i][j][2] = 0

        plt.imshow(image_copy)
        plt.show()


# Splits all image pixels into a list of pixels for each class
def split_pixels(images, masks, split_folder):

    # Files do not exist
    if not os.listdir(split_folder):

        # Split pixels
        barrel_pixels = []
        background_pixels = []
        for image, mask in zip(images, masks):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if mask[i][j]:
                        barrel_pixels.append(image[i][j])
                    else:
                        background_pixels.append(image[i][j])
        # Save files
        np.save(split_folder + 'red barrel.npy', barrel_pixels)
        np.save(split_folder + 'background.npy', background_pixels)

    # Load pixels per class
    pixels_per_class = [
        np.load(split_folder + 'red barrel.npy', allow_pickle=True),
        np.load(split_folder + 'background.npy', allow_pickle=True)
    ]

    # Return vector of pixels per class
    print("Splitting Pixels Complete " + str(time.time() - start))
    return pixels_per_class


# Convert scale
# Convert 8-bit into 6-bit (by dividing by 4)
# 8-bit (0-255) to 6-bit (0-63)
def scale_training_pixels(pixels_per_class, multiple):
    scaled_vectors = []
    for pixel_vector in pixels_per_class:
        scaled_vector = pixel_vector * multiple
        scaled_vectors.append(scaled_vector)

    print("Scaling Pixels Complete " + str(time.time() - start))
    return scaled_vectors


# Each class has a distribution defined by Mu vector of length 3, and Sigma a 3x3 matrix
def calculate_distributions(pixels_per_class, names):
    distributions = []
    for pixel_vector, name in zip(pixels_per_class, names):
        # Calculate mu and sigma (sigma must be transposed)
        mu = np.mean(pixel_vector, axis=0)
        sigma = np.cov(pixel_vector.T)
        distributions.append((mu, sigma, name))

    print("Calculating Distributions Complete " + str(time.time() - start))
    return distributions


# Plot each class's pixels and mu's in 3D space
def plot_pixels(pixels_per_class, distributions):

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Subsample pixels to plot
    samples = []
    for pixel_vector in pixels_per_class:
        sample = pixel_vector[np.random.randint(pixel_vector.shape[0], size=10000), :]
        samples.append(sample)

    # Plot classes by color
    colors = ['red', 'blue']
    for sample, color in zip(samples, colors):
        ax.scatter3D(sample[:, 0], sample[:, 1], sample[:, 2], c=color)

    # Plot mus
    colors = ['orange', 'green']
    for distribution, color in zip(distributions, colors):
        mu = distribution[0]
        ax.scatter3D(mu[0], mu[1], mu[2], c=color)

    # Plot legend
    names = []
    for distribution in distributions:
        names.append(distribution[2])
    for distribution in distributions:
        names.append(distribution[2] + ' mu')
    plt.legend(names)
    print("Plotting Complete " + str(time.time() - start))
    plt.show()


def calculate_likelihood(mu, sigma, pixel):
    # Calculate intermediaries
    v1 = np.subtract(pixel, mu).reshape((1, 3))
    v2 = v1.T

    # Calculate prediction
    top = exp(-0.5 * dot(dot(v1, inv(sigma)), v2))
    bottom = pow(2 * pi, 1.5) * sqrt(det(sigma))
    return top / bottom


# Calculates prior by taking the number of pixels in this class
# over the total number of pixels
def calculate_prior(pixels_per_class, index):
    total = 0
    length = 0
    for i, pixel_vector in enumerate(pixels_per_class):
        total += len(pixel_vector)
        if i == index:
            length = len(pixel_vector)

    prior = length / total
    return prior


# Calculates the normalization (denominator) for bayes rule
def calculate_normalization(distributions, pixels_per_class, pixel):

    norm_sum = 0
    for i, (distribution, pixel_vector) in enumerate(zip(distributions, pixels_per_class)):

        # Store variables
        mu = distribution[0]
        sigma = distribution[1]

        # Calculate each class likelihood*prior
        likelihood = calculate_likelihood(mu, sigma, pixel)
        prior = calculate_prior(pixels_per_class, i)
        norm_sum += likelihood * prior

    return norm_sum


# Calculate table for class in 6-bit
# Converted 8-bit RGB into 6-bit RGB (by dividing by 4)
# 8-bit (0-255) to 6-bit (0-63)
def create_table(table_file, distributions, dims, pixels_per_class, class_index):

    # Save distribution for this index
    distribution = distributions[class_index]

    # Initialize empty table
    table = np.empty(shape=dims)
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):

                # Store variables
                mu = distribution[0]
                sigma = distribution[1]
                pixel = np.array([i, j, k])

                # Calculate bayes components
                likelihood = calculate_likelihood(mu, sigma, pixel)
                prior = calculate_prior(pixels_per_class, class_index)
                normalization = calculate_normalization(distributions, pixels_per_class, pixel)
                table[i][j][k] = likelihood * prior / normalization

    np.save(table_file, table)
    print("Table Creation Complete " + str(time.time() - start))


def load_tables(table_folder, distributions, dimensions, pixels_per_class):

    # Create a table with each classes distribution if it does not exist
    tables = []
    for i, distribution in enumerate(distributions):
        name = distribution[2]
        table_file = table_folder + name + '.npy'
        if not os.path.exists(table_file):
            create_table(table_file, distributions, dimensions, pixels_per_class, i)

        table = np.load(table_file, allow_pickle=True)
        tables.append(table)

    print("Table Loading Complete " + str(time.time() - start))
    return tables


def predict_pixel(pixel, class_names, tables):

    # Get highest probability class
    max_name = None
    max_probability = 0

    # Save each class distribution into its own table
    for name, table in zip(class_names, tables):

        # The pixel belongs to the class with highest probability
        probability = table[pixel[0]][pixel[1]][pixel[2]]
        if probability > max_probability:
            max_probability = probability
            max_name = name

    return max_name


# Return bitmasks with red barrel labeled as 1
def predict_images(images, class_names, tables):

    # Save bitmasks for each image
    bitmasks = []
    for count, image in enumerate(images):

        bitmask = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                # Convert to 6-bit
                pixel = np.floor(image[i][j] / 4).astype(int)
                predicted_class = predict_pixel(pixel, class_names, tables)

                # Red barrel
                if predicted_class == 'red barrel':
                    bitmask[i][j] = 1

        # Save image bitmask
        bitmasks.append(bitmask)
        print('Image ' + str(count+1) + '/' + str(len(images)) + ' Complete ' + str(time.time() - start))

    print("Predicting Images Complete " + str(time.time() - start))
    return bitmasks


# Show all red barrel predicted pixels as red
def print_predictions(images, class_names, tables):

    for image in images:
        # Copy image to overwrite
        image_copy = image.copy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):

                # Convert to 6-bit
                pixel = np.floor(image[i][j] / 4).astype(int)
                predicted_class = predict_pixel(pixel, class_names, tables)

                # Red barrel
                if predicted_class == 'red barrel':
                    image_copy[i][j][0] = 255
                    image_copy[i][j][1] = 0
                    image_copy[i][j][2] = 0

        plt.imshow(image_copy)
        print("Pixel Prediction Printing Complete " + str(time.time() - start))
        plt.show()


# Draw box boundaries on each image
def draw_boundaries(image, regions, bounded_folder, name, distance_constant):
    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    # Save largest k regions
    k = 2
    areas = [region.area for region in regions]
    largest_indices = np.argsort(areas)[-k:]

    # Plot largest k regions
    for index in largest_indices:
        region = regions[index]

        # Plot only if similar dimensions to barrel
        if region.minor_axis_length == 0:
            continue

        ratio = region.major_axis_length / region.minor_axis_length
        if 1 < ratio < 3 and region.extent > .5:

            # Plot centroid
            y0, x0 = region.centroid
            ax.plot(x0, y0, '.g', markersize=15)

            print(x0)
            print(y0)

            # Plot bounding box
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-r', linewidth=2.5)

            # Plot distance
            dist = str(round(distance_constant / region.major_axis_length, 2))
            ax.annotate(dist, xy=region.centroid, xycoords='data', color='r')
            print(dist)
            print('\n')

    path = bounded_folder + name + '.png'
    plt.savefig(path, format='png')
    plt.close(fig)
    return cv2.imread(path)


# Use similar triangles to calculate distance to barrel
def calculate_distances(image, regions, distance_constant):

    distances = []
    for region in regions:
        distance = distance_constant / region.major_axis_length
        distances.append(distance)

    return distances


# Draw box boundaries on each image and determine distances
def label_images(bitmasks, images, distance_constant, image_folder, label_folder):

    #labeled_images = []
    for bitmask, image, name in zip(bitmasks, images, get_image_names(image_folder)):

        # Identify all regions
        regions = regionprops(label(bitmask))

        # Draw boundaries on image
        bounded_image = draw_boundaries(image, regions, label_folder, name, distance_constant)

        # Calculate distances
        #distances = calculate_distances(image, regions, distance_constant)
        #import pdb; pdb.set_trace()
        #labeled_image = write_distance_to_image(bounded_image, distances)
        #labeled_images.append(labeled_image)

    print("Image Labeling Complete " + str(time.time() - start))
    #return labeled_images


def save_images(images, folder):
    for image, name in zip(images, get_image_names(folder)):
        cv2.imwrite(folder + name, image)


def calculate_constant(base):

    # Store constant
    const = 0

    # Try reading in constant to calculate distances
    path = base + 'const.txt'
    if os.path.exists(path):
        with open(path) as f:
            const = f.readline()

    return const


def main():
    ########
    # Main #
    ########

    # Config
    distance_constant = 734
    class_names = ['red barrel', 'background']
    color_space = 'rgb'
    base = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project1'
    train_image_folder = base + '/train_images/'
    mask_folder = base + '/masks/'
    test_image_folder = base + '/test_images/'
    table_folder = base + '/tables_' + color_space + '/'
    label_folder = base + '/labeled_images/'
    split_folder = base + '/split/'
    bounded_folder = base + '/bounded_images/'

    """
        TRAINING 
    """

    # Load train images
    train_images = load_images(train_image_folder, color_space)

    # Load bit masks
    masks = load_masks(train_image_folder, mask_folder)

    # Calculate distance constant
    #distance_constant = calculate_constant(base)

    # Split pixels by class
    pixels_per_class = split_pixels(train_images, masks, split_folder)

    # Reduce pixel scale
    pixels_per_class = scale_training_pixels(pixels_per_class, 1/4)

    # Calculate distributions
    distributions = calculate_distributions(pixels_per_class, class_names)

    # Plot pixels and mus in 3D space
    #plot_pixels(pixels_per_class, distributions)

    # Load tables
    dimensions = [64, 64, 64]
    tables = load_tables(table_folder, distributions, dimensions, pixels_per_class)

    """
        TESTING 
    """

    # Load test images
    test_images = load_images(test_image_folder, color_space)

    # Predict test image pixels
    boundary_bitmasks = predict_images(test_images, class_names, tables)
    #print_predictions(test_images, class_names, tables)

    # Draw boxes and determine distances
    label_images(boundary_bitmasks, test_images, distance_constant, test_image_folder, label_folder)
    #save_images(labeled_images, label_folder)

if __name__ == "__main__":
    main()
