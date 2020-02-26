# Slice images and labels to 2d

import os
import numpy as np
import nibabel as nib
from imageio import imwrite
from skimage.transform import rescale

# Constants
OLD_PATH_LABELS_FILTERED = "../../data/Training/label_filtered/"
NEW_PATH_LABELS_FILTERED = "../../data/Training_2d/label_filtered/"
OLD_PATH_LABELS = "../../data/Training/label/"
NEW_PATH_LABELS = "../../data/Training_2d/label/"
OLD_PATH_IMG = "../../data/Training/img/"
NEW_PATH_IMG = "../../data/Training_2d/img/"


# Iterate through all files in filtered labels directory
for file_name in os.listdir(OLD_PATH_LABELS_FILTERED):
    # Load the image
    image = nib.load(OLD_PATH_LABELS_FILTERED + file_name)
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 147 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension and "label" prefix from file name.
        np.save(NEW_PATH_LABELS_FILTERED + file_name[5:-7] + "_" + str(index), slice)

# Repeat for unfiltered labels
for file_name in os.listdir(OLD_PATH_LABELS):
    # Load the image
    image = nib.load(OLD_PATH_LABELS + file_name)
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 147 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension and "label" prefix from file name.
        np.save(NEW_PATH_LABELS + file_name[5:-7] + "_" + str(index), slice)

# # Repeat for the actual images
# for file_name in os.listdir(OLD_PATH_IMG):
#     # Load the image
#     image = nib.load(OLD_PATH_IMG + file_name)
#     image_data = image.get_fdata()
#     # Iterate through the third dimension to create 147 2d slices of size 512x512
#     for index in range(image_data.shape[2]):
#         # Get the current slice
#         slice = image_data[:, :, index]
#         # Convert to int16
#         slice = slice.astype(np.int16)
#         # Save as numpy array. Exclude extension and "img" prefix from file name.
#         np.save(NEW_PATH_IMG + file_name[3:-7] + "_" + str(index), slice)
