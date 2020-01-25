# Imports for Pytorch
from __future__ import print_function
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from skimage import io, transform

"""
Generate class labels corresponding to windows for the x and y values to
turn this into classification problem. After predicting labels, we will use
the center of each window to predict an x and y coordinate for the object.
:return: None
"""
# Load in labels file and rename column names.
labels_df = pd.read_csv("../data/labels/labels.txt", sep=" ", header=None)
labels_df.columns = ["file_name", "x", "y"]

# Create new row with the class for the x coordinate. We have 20 classes representing a division of the
# x space into 15 equally wide regions.
labels_df["x_class"] = (np.floor(labels_df["x"] * WINDOWS)).astype(int)
# Create new row with the class for the x coordinate. We have 20 classes representing a division of the
# x space into 15 equally wide regions.
labels_df["y_class"] = (np.floor(labels_df["y"] * WINDOWS)).astype(int)
# Drop original labels
labels_df = labels_df.drop(columns=["x", "y"])

# Get the rows corresponding to training and validation sets.
val_labels_df = labels_df[labels_df["file_name"].isin(VALIDATION_NAMES)]
train_labels_df = labels_df[~labels_df["file_name"].isin(VALIDATION_NAMES)]
# Store the label names separately
val_labels_df.to_csv("../data/labels/validation_labels15.txt", sep=" ", index=False, header=False)
train_labels_df.to_csv("../data/labels/train_labels15.txt", sep=" ", index=False, header=False)