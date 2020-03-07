# Slice images into 2d slices for 2d networks.
# Create filtered version of labels with only spleen labels.

import nibabel as nib
import numpy as np
import os
from skimage.transform import resize

# Constants for path names
NEW_TRAIN_LABELS_FILTERED = "../../../data/Train/affine_fixed/label_cropped_filtered/"
OLD_TRAIN_LABELS = "../../../data/Train/affine_fixed/label_registered/"
NEW_TRAIN_LABELS = "../../../data/Train/affine_fixed/label_cropped/"
OLD_TRAIN_IMG = "../../../data/Train/affine_fixed/img_registered/"
NEW_TRAIN_IMG = "../../../data/Train/affine_fixed/img_cropped/"
NEW_VAL_LABELS_FILTERED = "../../../data/Val/affine_fixed/label_cropped_filtered/"
OLD_VAL_LABELS = "../../../data/Val/affine_fixed/label_registered/"
NEW_VAL_LABELS = "../../../data/Val/affine_fixed/label_cropped/"
OLD_VAL_IMG = "../../../data/Val/affine_fixed/img_registered/"
NEW_VAL_IMG = "../../../data/Val/affine_fixed/img_cropped/"
# Start and end indices on z axis to reslice, since most slices do not have spleen
Z_START = 75
Z_END = 145

# Start and end indices on x axis to reslice, since most slices do not have spleen
X_START = 288
X_END = 512
Y_START = 110
Y_END = 334

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
        if index < image_data.shape[2]:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index]
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
    # Get version of labels with only spleen labels (label 1).
    spleen = np.where(image_data == 1, 1, 0)
    # Calculate original sum of spleen labels and increment running total
    original_spleen_labels = np.sum(spleen)
    old_train_sum += original_spleen_labels
    # Calculate new sum of spleen labels. Start at 0 and add at each slice
    new_spleen_labels = 0
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Check that we have not reached the end of the image
        if index < image_data.shape[2]:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index]
            # Convert to uint8
            slice = slice.astype(np.uint8)
            # Save as numpy array. Exclude extension prefix from file name.
            np.save(NEW_TRAIN_LABELS + file_name[:-7] + "_" + str(index), slice)
            # Save version of labels with only spleen labels (label 1).
            spleen_slice = np.where(slice==1, 1, 0)
            spleen_slice = spleen_slice.astype(np.uint8)
            np.save(NEW_TRAIN_LABELS_FILTERED + file_name[:-7] + "_" + str(index), spleen_slice)
            # Increment new sum of spleen labels
            new_spleen_labels += np.sum(spleen_slice)
    # Increment sum of all spleen labels
    new_train_sum += new_spleen_labels
print("Original Training Number of spleen labels: ", old_train_sum)
print("New Training Number of spleen labels: ", new_train_sum)
print("Percentage of Training Spleen Labels Retained: ", new_train_sum / old_train_sum)

print("Percentage of Training Labels that is spleen: ", new_train_sum / (224*224*60*24))
print("Original percentage of Training Labels that is spleen: ", new_train_sum / (512*512*163*24))

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
        if index < image_data.shape[2]:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index]
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
    # Get version of labels with only spleen labels (label 1).
    spleen = np.where(image_data == 1, 1, 0)
    # Calculate original sum of spleen labels
    original_spleen_labels = np.sum(spleen)
    old_val_sum += original_spleen_labels
    # Calculate new sum of spleen labels. Start at 0 and add at each slice
    new_spleen_labels = 0
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
        # Check that we have not reached the end of the image
        if index < image_data.shape[2]:
            # Get the current slice
            slice = image_data[X_START:X_END, Y_START:Y_END, index]
            # Convert to uint8
            slice = slice.astype(np.uint8)
            # Save as numpy array. Exclude extension prefix from file name.
            np.save(NEW_VAL_LABELS + file_name[:-7] + "_" + str(index), slice)
            # Save version of labels with only spleen labels (label 1).
            spleen_slice = np.where(slice==1, 1, 0)
            spleen_slice = spleen_slice.astype(np.uint8)
            np.save(NEW_VAL_LABELS_FILTERED + file_name[:-7] + "_" + str(index), spleen_slice)
            # Increment new sum of spleen labels
            new_spleen_labels += np.sum(spleen_slice)
    # Update sum of new spleen labels
    new_val_sum += new_spleen_labels
print("Original Validation Number of spleen labels: ", old_val_sum)
print("New Validation Number of spleen labels: ", new_val_sum)
print("Percentage of Spleen Labels Retained: ", new_val_sum / old_val_sum)
print("Percentage of Val Labels that is spleen: ", new_val_sum / (224*224*60*6))
print("Original percentage of Val Labels that is spleen: ", new_val_sum / (512*512*163*6))