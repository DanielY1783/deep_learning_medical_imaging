# Rescale registered images to -1 to 1 range

import nibabel as nib
import numpy as np
import os

# Constants for path names
OLD_TRAIN_IMG = "../../../data/Train/affine/img_registered_no_resize/"
NEW_TRAIN_IMG = "../../../data/Train/affine/img_rescaled/"
OLD_VAL_IMG = "../../../data/Val/affine/img_registered_no_resize/"
NEW_VAL_IMG = "../../../data/Val/affine/img_rescaled/"

# First for training set
# Iterate through all the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Set zero values to -1000, since values that are exactly zero should represent padded background
    image_data = np.where(image_data == 0.0, -1000, image_data)
    # Divide by 1000 to get in -1 to 1 range
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
    # Set zero values to -1000, since values that are exactly zero should represent padded background
    image_data = np.where(image_data == 0.0, -1000, image_data)
    # Divide by 1000 to get in -1 to 1 range
    image_data = image_data / 1000.0
    # Save the image
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_VAL_IMG + file_name)

