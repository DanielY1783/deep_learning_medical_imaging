# Perform 80/20 train-val split first

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import shutil

# Constants
ORIGINAL_LABELS = "../../data/Original_Training/label/"
TRAIN_LABELS = "../../data/Train/label/"
VAL_LABELS = "../../data/Val/label/"
ORIGINAL_IMG = "../../data/Original_Training/img/"
TRAIN_IMG = "../../data/Train/img/"
VAL_IMG = "../../data/Val/img/"

# Get list of all the volume names
volumes = []
for file_name in os.listdir(ORIGINAL_LABELS):
    volumes.append(file_name[5:])

# 80/20 train-val split with scikit learn
train, val = train_test_split(volumes, test_size=0.2)

# Copy image and label to new directories for train and validation
for volume in train:
    # Copy image
    shutil.copy(ORIGINAL_IMG + "img" + volume, TRAIN_IMG + volume)
    # Copy label
    shutil.copy(ORIGINAL_LABELS + "label" + volume, TRAIN_LABELS + volume)

for volume in val:
    # Copy image
    shutil.copy(ORIGINAL_IMG + "img" + volume, VAL_IMG + volume)
    # Copy label
    shutil.copy(ORIGINAL_LABELS + "label" + volume, VAL_LABELS + volume)