# Slice images into 2d slices for 2d networks.
# Create filtered version of labels with only spleen labels.

import nibabel as nib
import numpy as np
import os
from skimage.transform import resize

# Constants for path names
NEW_TRAIN_LABELS_FILTERED = "../../data/train_heavy_crop/label_filtered/"
OLD_TRAIN_LABELS = "../../data/Train/label_registered/"
NEW_TRAIN_LABELS = "../../data/train_heavy_crop/label/"
OLD_TRAIN_IMG = "../../data/Train/img_rescaled/"
NEW_TRAIN_IMG = "../../data/train_heavy_crop/img/"
NEW_VAL_LABELS_FILTERED = "../../data/val_heavy_crop/label_filtered/"
OLD_VAL_LABELS = "../../data/Val/label_registered/"
NEW_VAL_LABELS = "../../data/val_heavy_crop/label/"
OLD_VAL_IMG = "../../data/Val/img_rescaled/"
NEW_VAL_IMG = "../../data/val_heavy_crop/img/"
# Start and end indices on z axis to reslice, since most slices do not have spleen
Z_START = 37
Z_END = 63

# Start and end indices on x axis to reslice, since most slices do not have spleen
X_START = 137
X_END = 210
Y_START = 50
Y_END = 150

# First for training set
# Iterate through all the actual images
for file_name in os.listdir(OLD_TRAIN_IMG):
    # Load the image
    image = nib.load(OLD_TRAIN_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
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
print("Percentage of Spleen Labels Retained: ", new_train_sum / old_train_sum)

print("Percentage of image that is spleen: ", new_train_sum / (83*100*26*23))
print("Original percentage of image that is spleen: ", new_train_sum / (224*224*26*23))

# Repeat for Validation Set
# Iterate through all the actual images
for file_name in os.listdir(OLD_VAL_IMG):
    # Load the image
    image = nib.load(OLD_VAL_IMG + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    # Iterate through the third dimension to create 2d slices
    for index in range(Z_START, Z_END + 1):
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