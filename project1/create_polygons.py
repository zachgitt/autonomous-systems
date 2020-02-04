import os
import cv2
import logging

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib import pyplot as plt
from roipoly import MultiRoi

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Create image
img = None
file = None
folder_in = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project1/train_images/'
for filename in os.listdir(folder_in):
    # Default reads in as BGR (blue-green-red encoding)
    img_bgr = cv2.imread(os.path.join(folder_in, filename))
    if img_bgr is not None:
        # Convert to HSV
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img = img_gray
        file = filename
        break


# Show the image
fig = plt.figure()
plt.imshow(img, interpolation='nearest', cmap="Greys")
plt.title("Click on the button to add a new ROI")

# Draw multiple ROIs
# NOTE: If you create more ROIs than the number of classes, only the
#       last ROI will be saved
multiroi_named = MultiRoi(roi_names=['red barrel-1', 'red barrel-2'])

# Store mask
masks = []

# Draw all ROIs
plt.imshow(img, interpolation='nearest', cmap="Greys")
roi_names = []
for name, roi in multiroi_named.rois.items():
    roi.display_roi()
    roi.display_mean(img)
    roi_names.append(name)
    masks.append(roi.get_mask(img))
plt.legend(roi_names, bbox_to_anchor=(1.2, 1.05))
plt.show()

# Print bit masks
size = 0
print(file)
for mask in masks:
    size = 0
    for row in mask:
        for cell in row:
            if cell:
                size += 1
    print("Size: " + str(size))

# Union masks
mask = np.logical_or(masks[0], masks[1])

# Save bitmask
folder_out = '/Users/zacharygittelman/Documents/repos/autonomous-systems/project1/masks/'
path = folder_out + file[:-4] # Remove .png
if not os.path.exists(path):
    np.save(path, mask)

# Load bitmask
size_loaded = 0
loaded_mask = np.load(path + '.npy', allow_pickle=True)
for r in loaded_mask:
    for c in r:
        if c:
            size_loaded += c
print("LOADED SIZE: " + str(size_loaded))
