# Generate binary labels for 0/1 for only spleen

import os
import numpy as np
import nibabel as nib

# Constants
OLD_PATH_TRAIN = "../../data/Train/label/"
NEW_PATH_TRAIN = "../../data/Train/label_filtered/"
OLD_PATH_VAL = "../../data/Val/label/"
NEW_PATH_VAL = "../../data/Val/label_filtered/"

# Iterate through all files in labels directory and create binary 0/1 labels for spleen
# for the training set
for file_name in os.listdir(OLD_PATH_TRAIN):
    # Load in nii file
    image = nib.load(OLD_PATH_TRAIN + file_name)
    image_data = image.get_fdata()
    # Filter for 1 label for spleen
    image_data_filtered = np.where(image_data==1, 1.0, 0.0)
    image_filtered = nib.Nifti1Image(image_data_filtered, image.affine)
    # Save the image
    nib.save(image_filtered, NEW_PATH_TRAIN + file_name)

# Repeat for validation set
for file_name in os.listdir(OLD_PATH_VAL):
    # Load in nii file
    image = nib.load(OLD_PATH_VAL + file_name)
    image_data = image.get_fdata()
    # Filter for 1 label for spleen
    image_data_filtered = np.where(image_data==1, 1.0, 0.0)
    image_filtered = nib.Nifti1Image(image_data_filtered, image.affine)
    # Save the image
    nib.save(image_filtered, NEW_PATH_VAL + file_name)