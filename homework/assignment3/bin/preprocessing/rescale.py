# Then rescale image values to 0-1 scale by first subtracting the minimum value to get the minimum value to
# 0, and then divide by the maximum value to get to 0-1 range. Then subtract 0.45 as mean and 0.225 as standard
# deviation for imagenet standardization

import nibabel as nib
import numpy as np
import os

# Constants for path names
OLD_TRAIN_IMG = "../../data/Train/img/"
NEW_TRAIN_IMG = "../../data/Train/img_rescale/"
OLD_VAL_IMG = "../../data/Val/img/"
NEW_VAL_IMG = "../../data/Val/img_rescale/"

# First for training set
# Iterate through all the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Set negative values to zero
    image_data = np.where(image_data < 0, 0, image_data)
    # Calculate the mean and standard deviation
    mean = np.mean(image_data)
    std = np.std(image_data)
    # Subtract mean and divide by standard deviation to normalize
    image_data = (image_data - mean) / std
    # # Calculate the minimum value of the data and subtract all values by that value to get min value to 0
    # min = np.amin(image_data)
    # image_data = image_data - min
    # # Calculate the new maximum value and divide by maximum value to get in 0-1 range
    # max = np.amax(image_data)
    # image_data = image_data / max
    # # Subtract 0.45 as mean for imagenet normalization
    # image_data = image_data - 0.45
    # # Divide by 0.225 as standard deviation for imagenet normalization
    # image_data = image_data / 0.225
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_TRAIN_IMG + file_name)

# Repeat for Validation Set
# Iterate through all the actual images
for file_name in os.listdir(OLD_VAL_IMG):
    # Load the image
    image = nib.load(OLD_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Set negative values to zero
    image_data = np.where(image_data < 0, 0, image_data)
    # Calculate the mean and standard deviation
    mean = np.mean(image_data)
    std = np.std(image_data)
    # Subtract mean and divide by standard deviation to normalize
    image_data = (image_data - mean) / std


    # # Calculate the minimum value of the data and subtract all values by that value to get min value to 0
    # min = np.amin(image_data)
    # image_data = image_data - min
    # # Calculate the new maximum value and divide by maximum value to get in 0-1 range
    # max = np.amax(image_data)
    # image_data = image_data / max
    # # Subtract 0.45 as mean for imagenet normalization
    # image_data = image_data - 0.45
    # # Divide by 0.225 as standard deviation for imagenet normalization
    # image_data = image_data / 0.225
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_VAL_IMG + file_name)

