# Resize the image to 224x224x70 to use with torchvision models, which require 224x224 input
# Then rescale image values to 0-1 scale by first subtracting the minimum value to get the minimum value to
# 0, and then divide by the maximum value to get to 0-1 range.

import nibabel as nib
import numpy as np
import os
from skimage.transform import resize

# Constants for path names
OLD_TRAIN_LABELS_FILTERED = "../../data/Train/label_filtered/"
NEW_TRAIN_LABELS_FILTERED = "../../data/Train224_2d/label_filtered_2d/"
OLD_TRAIN_LABELS = "../../data/Train/label/"
NEW_TRAIN_LABELS = "../../data/Train224_2d/label_2d/"
OLD_TRAIN_IMG = "../../data/Train/img/"
NEW_TRAIN_IMG = "../../data/Train224_2d/img_2d/"
OLD_VAL_LABELS_FILTERED = "../../data/Val/label_filtered/"
NEW_VAL_LABELS_FILTERED = "../../data/Val224_2d/label_filtered_2d/"
OLD_VAL_LABELS = "../../data/Val/label/"
NEW_VAL_LABELS = "../../data/Val224_2d/label_2d/"
OLD_VAL_IMG = "../../data/Val/img/"
NEW_VAL_IMG = "../../data/Val224_2d/img_2d/"

# First for training set
# Iterate through all the actual images
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
        # Convert to float16
        slice = slice.astype(np.float16)
        # Resize to 224x224
        slice = resize(slice, (224, 224) ,preserve_range=True, anti_aliasing=False)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_IMG + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels
for file_name in os.listdir(OLD_TRAIN_LABELS):
    # Load the image
    image = nib.load(OLD_TRAIN_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Resize to 224x224
        slice = resize(slice, (224, 224) ,preserve_range=True, anti_aliasing=False)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_LABELS + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels filtered only for spleen
for file_name in os.listdir(OLD_TRAIN_LABELS_FILTERED):
    # Load the image
    image = nib.load(OLD_TRAIN_LABELS_FILTERED + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Resize to 224x224
        slice = resize(slice, (224, 224) ,preserve_range=True, anti_aliasing=False)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_TRAIN_LABELS_FILTERED + file_name[:-7] + "_" + str(index), slice)

# Repeat for Validation Set
# Iterate through all the actual images
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
        # Resize to 224x224
        slice = resize(slice, (224, 224) ,preserve_range=True, anti_aliasing=False)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_IMG + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels
for file_name in os.listdir(OLD_VAL_LABELS):
    # Load the image
    image = nib.load(OLD_VAL_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Resize to 224x224
        slice = resize(slice, (224, 224) ,preserve_range=True, anti_aliasing=False)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_LABELS + file_name[:-7] + "_" + str(index), slice)

# Iterate through the labels filtered only for spleen
for file_name in os.listdir(OLD_VAL_LABELS_FILTERED):
    # Load the image
    image = nib.load(OLD_VAL_LABELS_FILTERED + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices of size 512x512
    for index in range(image_data.shape[2]):
        # Get the current slice
        slice = image_data[:, :, index]
        # Convert to uint8
        slice = slice.astype(np.uint8)
        # Resize to 224x224
        slice = resize(slice, (224, 224) ,preserve_range=True, anti_aliasing=False)
        # Save as numpy array. Exclude extension prefix from file name.
        np.save(NEW_VAL_LABELS_FILTERED + file_name[:-7] + "_" + str(index), slice)