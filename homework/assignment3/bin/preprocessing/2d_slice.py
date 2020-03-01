# Slice images and labels to 2d

import os
import numpy as np
import nibabel as nib

# Constants for path names
OLD_TRAIN_LABELS_FILTERED = "../../data/Train/label_filtered/"
NEW_TRAIN_LABELS_FILTERED = "../../data/Train/label_filtered_2d/"
OLD_TRAIN_LABELS = "../../data/Train/label/"
NEW_TRAIN_LABELS = "../../data/Train/label_2d/"
OLD_TRAIN_IMG = "../../data/Train/img/"
NEW_TRAIN_IMG = "../../data/Train/img_2d/"
OLD_VAL_LABELS_FILTERED = "../../data/Val/label_filtered/"
NEW_VAL_LABELS_FILTERED = "../../data/Val/label_filtered_2d/"
OLD_VAL_LABELS = "../../data/Val/label/"
NEW_VAL_LABELS = "../../data/Val/label_2d/"
OLD_VAL_IMG = "../../data/Val/img/"
NEW_VAL_IMG = "../../data/Val/img_2d/"

# First for training set
# Iterate through all files in filtered labels directory
for file_name in os.listdir(OLD_TRAIN_LABELS_FILTERED):
    # Load the image
    image = nib.load(OLD_TRAIN_LABELS_FILTERED + file_name)
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_LABELS_FILTERED + file_name[:-7] + "_" + str(index), slice)

# Repeat for unfiltered labels
for file_name in os.listdir(OLD_TRAIN_LABELS):
    # Load the image
    image = nib.load(OLD_TRAIN_LABELS + file_name)
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_LABELS + file_name[:-7] + "_" + str(index), slice)

# Repeat for the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Calculate the minimum value of the data and subtract all values by that value to get min value to 0
    min = np.amin(image_data)
    image_data = image_data - min
    # Calculate the new maximum value and divide by maximum value to get in 0-1 range
    max = np.amax(image_data)
    image_data = image_data / max
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to int16
        slice = slice.astype(np.float16)
        # Save as numpy array. Exclude extension from file name.
        np.save(NEW_TRAIN_IMG + file_name[:-7] + "_" + str(index), slice)

# Repeat for validation set
# Iterate through all files in filtered labels directory
for file_name in os.listdir(OLD_VAL_LABELS_FILTERED):
    # Load the image
    image = nib.load(OLD_VAL_LABELS_FILTERED + file_name)
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_LABELS_FILTERED + file_name[:-7] + "_" + str(index), slice)

# Repeat for unfiltered labels
for file_name in os.listdir(OLD_VAL_LABELS):
    # Load the image
    image = nib.load(OLD_VAL_LABELS + file_name)
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_LABELS + file_name[:-7] + "_" + str(index), slice)

# Repeat for the actual images
for file_name in os.listdir(OLD_VAL_IMG):
    # Load the image
    image = nib.load(OLD_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Calculate the minimum value of the data and subtract all values by that value to get min value to 0
    min = np.amin(image_data)
    image_data = image_data - min
    # Calculate the new maximum value and divide by maximum value to get in 0-1 range
    max = np.amax(image_data)
    image_data = image_data / max
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to float16
        slice = slice.astype(np.float16)
        # Save as numpy array. Exclude extension from file name.
        np.save(NEW_VAL_IMG + file_name[:-7] + "_" + str(index), slice)
