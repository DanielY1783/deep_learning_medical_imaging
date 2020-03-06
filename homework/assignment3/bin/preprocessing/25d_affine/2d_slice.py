# Slice images into 2d slices for 2d networks.
# Create filtered version of labels with only spleen labels.

import nibabel as nib
import numpy as np
import os
from skimage.transform import resize

# Constants for path names
NEW_TRAIN_LABELS_FILTERED = "../../../data/Train/25d_affine/label_filtered/"
OLD_TRAIN_LABELS = "../../../data/Train/affine/label_registered_no_resize/"
NEW_TRAIN_LABELS = "../../../data/Train/25d_affine/label/"
OLD_TRAIN_IMG = "../../../data/Train/affine/img_registered_no_resize/"
NEW_TRAIN_IMG = "../../../data/Train/25d_affine/img/"
NEW_VAL_LABELS_FILTERED = "../../../data/Val/25d_affine/label_filtered/"
OLD_VAL_LABELS = "../../../data/Val/affine/label_registered_no_resize/"
NEW_VAL_LABELS = "../../../data/Val/25d_affine/label/"
OLD_VAL_IMG = "../../../data/Val/affine/img_registered_no_resize/"
NEW_VAL_IMG = "../../../data/Val/25d_affine/img/"
# Start and end indices on z axis to reslice, since most slices do not have spleen
Z_START = 80
Z_END = 140

# Start and end indices on x axis to reslice, since most slices do not have spleen
X_START = 276
X_END = 500
Y_START = 100
Y_END = 324

# First for training set
# Iterate through all the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Fix zero values added by registration
    image_data = np.where(image_data == 0.0, -1000, image_data)
    # Divide by 1000 to normalize
    image_data = image_data / 1000.0
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Check that we have not reached the end of the image
        if index < image_data.shape[2] - 2:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index-1: index+2]
            # Convert to float16
            slice = slice.astype(np.float16)
            # Save as numpy array. Exclude extension prefix from file name.
            np.save(NEW_TRAIN_IMG + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels
# Store original and new sum of spleen labels for all training images
old_train_sum = 0
new_train_sum = 0
for file_name in os.listdir(OLD_TRAIN_LABELS):
    # Load the image
    image = nib.load(OLD_TRAIN_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Check that we have not reached the end of the image
        if index < image_data.shape[2] - 2:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index-1: index+2]
            # Convert to uint8
            slice = slice.astype(np.uint8)
            # Save as numpy array. Exclude extension prefix from file name.
            np.save(NEW_TRAIN_LABELS + file_name[:-7] + "_" + str(index), slice)
            # Save version of labels with only spleen labels (label 1).
            spleen_slice = np.where(slice==1, 1, 0)
            spleen_slice = spleen_slice.astype(np.uint8)
            np.save(NEW_TRAIN_LABELS_FILTERED + file_name[:-7] + "_" + str(index), spleen_slice)

# Repeat for Validation Set
# Iterate through all the actual images
for file_name in os.listdir(OLD_VAL_IMG):
    # Load the image
    image = nib.load(OLD_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Fix zero values added by registration
    image_data = np.where(image_data == 0.0, -1000, image_data)
    # Divide by 1000 to normalize
    image_data = image_data / 1000.0
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Check that we have not reached the end of the image
        if index < image_data.shape[2] - 2:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index-1: index+2]
            # Convert to float16
            slice = slice.astype(np.float16)
            # Save as numpy array. Exclude extension prefix from file name.
            np.save(NEW_VAL_IMG + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels
# Store original and new sum of spleen labels for all validation images
old_val_sum = 0
new_val_sum = 0
for file_name in os.listdir(OLD_VAL_LABELS):
    # Load the image
    image = nib.load(OLD_VAL_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Check that we have not reached the end of the image
        if index < image_data.shape[2] - 2:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index-1: index+2]
            # Convert to uint8
            slice = slice.astype(np.uint8)
            # Save as numpy array. Exclude extension prefix from file name.
            np.save(NEW_VAL_LABELS + file_name[:-7] + "_" + str(index), slice)
            # Save version of labels with only spleen labels (label 1).
            spleen_slice = np.where(slice==1, 1, 0)
            spleen_slice = spleen_slice.astype(np.uint8)
            np.save(NEW_VAL_LABELS_FILTERED + file_name[:-7] + "_" + str(index), spleen_slice)