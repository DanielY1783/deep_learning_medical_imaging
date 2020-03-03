# Resize all images to [224, 224, 70] for registration, and rescale to -1 to 1 range
# Code for resizing from https://github.com/Yt-trium/nii-resize

import itertools
import nibabel as nib
import numpy as np
import os

# Constants for path names
OLD_TRAIN_IMG = "../../data/Train/img/"
NEW_TRAIN_IMG = "../../data/Train/img_resize/"
OLD_TRAIN_LABEL = "../../data/Train/label/"
NEW_TRAIN_LABEL = "../../data/Train/label_resize/"
OLD_VAL_IMG = "../../data/Val/img/"
NEW_VAL_IMG = "../../data/Val/img_resize/"
OLD_VAL_LABEL = "../../data/Val/label/"
NEW_VAL_LABEL = "../../data/Val/label_resize/"
NEW_SIZE = [224, 224, 70]

# Iterate through all pairs of training and validation images and labels.
for old_path, new_path in [(OLD_TRAIN_IMG, NEW_TRAIN_IMG), (OLD_VAL_IMG, NEW_VAL_IMG),
                           (OLD_TRAIN_LABEL, NEW_TRAIN_LABEL), (OLD_VAL_LABEL, NEW_VAL_LABEL)]:
    for file_name in os.listdir(old_path):
        # Load the image
        image = nib.load(old_path + file_name)
        # Get the array of values
        image_data = image.get_fdata()

        # Use resizing algorithm from https://github.com/Yt-trium/nii-resize
        delta_x = image_data.shape[0] / NEW_SIZE[0]
        delta_y = image_data.shape[1] / NEW_SIZE[1]
        delta_z = image_data.shape[2] / NEW_SIZE[2]

        new_data = np.zeros((NEW_SIZE[0], NEW_SIZE[1], NEW_SIZE[2]))

        for x, y, z in itertools.product(range(NEW_SIZE[0]),
                                         range(NEW_SIZE[1]),
                                         range(NEW_SIZE[2])):
            new_data[x][y][z] = image_data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

        # Flip the x axis data to get original coordinates
        new_data = np.flip(new_data, axis=1)

        # Save the image.
        image = nib.Nifti1Image(new_data, image.affine)
        nib.save(image, new_path + file_name)

# Rescale the images
# First for training set
for file_name in os.listdir(NEW_TRAIN_IMG):
    # Load the image
    image = nib.load(NEW_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Add by 1000 and then divide by 1000 get values in 0 to 2 range
    image_data = (image_data + 1000) / 1000.0
    # Save the image
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_TRAIN_IMG + file_name)

# Rescale the validation set
for file_name in os.listdir(NEW_VAL_IMG):
    # Load the image
    image = nib.load(NEW_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Add by 1000 and then divide by 1000 get values in 0 to 2 range
    image_data = (image_data + 1000) / 1000.0
    # Save the image
    image = nib.Nifti1Image(image_data, image.affine)
    nib.save(image, NEW_VAL_IMG + file_name)