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
    # Divide by 1000 to get values in -1 to 1 range
    image_data = image_data / 1000.0
    # Save the image
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_TRAIN_IMG + file_name)

# Repeat for Validation Set
# Iterate through all the actual images
for file_name in os.listdir(OLD_VAL_IMG):
    # Load the image
    image = nib.load(OLD_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Divide by 1000 to get values in -1 to 1 range
    image_data = image_data / 1000.0
    # Save the image
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_VAL_IMG + file_name)

