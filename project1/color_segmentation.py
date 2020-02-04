import cv2
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from roipoly import MultiRoi
import time

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


# Read masks
def load_masks(image_folder, mask_folder):

    # Load masks for each image
    masks = []
    for image_file in os.listdir(image_folder):

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
def split_pixels(images, masks):
    barrel_pixels = []
    background_pixels = []
    for image, mask in zip(images, masks):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mask[i][j]:
                    barrel_pixels.append(image[i][j])
                else:
                    background_pixels.append(image[i][j])

    print("Splitting Pixels Complete " + str(time.time() - start))
    return np.array(barrel_pixels), np.array(background_pixels)


# Each class has a distribution defined by Mu vector of length 3, and Sigma a 3x3 matrix
def calculate_distributions(pixel_vectors):
    distributions = []
    for pixels in pixel_vectors:
        # Calculate mu and sigma (sigma must be transposed)
        mu = np.mean(pixels, axis=0)
        sigma = np.cov(pixels.T)
        distributions.append((mu, sigma))

    print("Calculating Distributions Complete" + str(time.time() - start))
    return distributions


# Plot each class's pixels and mu's in 3D space
def plot_pixels(pixels, distributions):

    # Initialize figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot classes color
    colors = ['r', 'b']
    for pixel_vector, distribution, color in zip(pixels, distributions, colors):
        mu = distribution[0]
        ax.scatter3D(pixel_vector[0], pixel_vector[1], pixel_vector[2], c=color)

    red_dot, = plt.plot(5, "ro", markersize=5)
    blue_dot, = plt.plot(5, "bo", markersize=5)
    plt.legend(['red barrel', 'background'], bbox_to_anchor=(1.2, 1.05))
    plt.show()


def main():
    ########
    # Main #
    ########

    # Config
    base = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project1'
    train_image_folder = base + '/train_images/'
    mask_folder = base + '/masks/'
    test_image_folder = base + '/test_images/'

    # Load train_images
    train_images = load_images(train_image_folder, 'hsv')

    # Load bitmasks
    masks = load_masks(train_image_folder, mask_folder)

    # Split pixels by class
    barrel_pixels, background_pixels = split_pixels(train_images, masks)

    # Calculate distributions
    # TODO: mu and sigma may need to be casted to integers
    distributions = calculate_distributions([barrel_pixels, background_pixels])

    # Plot pixels and mu's in 3D space
    plot_pixels([barrel_pixels, background_pixels], distributions)

    # Load test train_images
    test_images = load_images(test_image_folder, 'hsv')

    # Predict test image pixels

    # Draw boundary


if __name__ == "__main__":
    main()
