# Rescale original images to positive values by adding 1000

import nibabel as nib
import numpy as np
import os

# Constants for path names
OLD_TRAIN_IMG = "../../data/Train/img/"
NEW_TRAIN_IMG = "../../data/Train/img_rescaled/"
OLD_VAL_IMG = "../../data/Val/img/"
NEW_VAL_IMG = "../../data/Val/img_rescaled/"

# First for training set
# Iterate through all the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Add by 1000 to get positive values
    image_data = image_data + 1000.0
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
    # Add by 1000 to get positive values
    image_data = image_data + 1000.0
    # Save the image
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_VAL_IMG + file_name)

