# Slice images into 2d slices for 2d networks.
# Create filtered version of labels with only spleen labels.

import nibabel as nib
import numpy as np
import os
from skimage.transform import resize

# Constants for path names
NEW_TRAIN_LABELS_FILTERED = "../../data/Train2d/label_filtered/"
OLD_TRAIN_LABELS = "../../data/Train/label_registered/"
NEW_TRAIN_LABELS = "../../data/Train2d/label/"
OLD_TRAIN_IMG = "../../data/Train/img_rescaled/"
NEW_TRAIN_IMG = "../../data/Train2d/img/"
NEW_VAL_LABELS_FILTERED = "../../data/Val2d/label_filtered/"
OLD_VAL_LABELS = "../../data/Val/label_registered/"
NEW_VAL_LABELS = "../../data/Val2d/label/"
OLD_VAL_IMG = "../../data/Val/img_rescaled/"
NEW_VAL_IMG = "../../data/Val2d/img/"
# Start and end indices on z axis to reslice, since most slices do not have spleen
Z_START = 38
Z_END = 60

# First for training set
# Iterate through all the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to float16
        slice = slice.astype(np.float16)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_IMG + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels
for file_name in os.listdir(OLD_TRAIN_LABELS):
    # Load the image
    image = nib.load(OLD_TRAIN_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_LABELS + file_name[:-7] + "_" + str(index), slice)
        # Save version of labels with only spleen labels (label 1).
        spleen = np.where(slice==1, 1, 0)
        spleen = spleen.astype(np.uint8)
        np.save(NEW_TRAIN_LABELS_FILTERED + file_name[:-7] + "_" + str(index), spleen)

# Repeat for Validation Set
# Iterate through all the actual images
for file_name in os.listdir(OLD_VAL_IMG):
    # Load the image
    image = nib.load(OLD_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to float16
        slice = slice.astype(np.float16)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_IMG + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels
for file_name in os.listdir(OLD_VAL_LABELS):
    # Load the image
    image = nib.load(OLD_VAL_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_LABELS + file_name[:-7] + "_" + str(index), slice)
        # Save version of labels with only spleen labels (label 1).
        spleen = np.where(slice==1, 1, 0)
        spleen = spleen.astype(np.uint8)
        np.save(NEW_VAL_LABELS_FILTERED + file_name[:-7] + "_" + str(index), spleen)
