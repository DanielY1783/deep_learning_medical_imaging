# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constant for names of validation files
VALIDATION_NAMES = ["111.jpg", "112.jpg", "113.jpg", "114.jpg", "115.jpg",
                    "116.jpg", "117.jpg", "118.jpg", "119.jpg", "125.jpg"]

# Load in labels file and rename column names.
labels_df = pd.read_csv("../data/labels/labels.txt", sep=" ", header=None)
labels_df.columns = ["file_name", "x", "y"]

# Create new row with the class for the x coordinate. We have 20 classes representing a division of the
# x space into 20 equally wide regions.
labels_df["x_class"] = (np.floor(labels_df["x"] / 0.05)).astype(int)
# Create new row with the class for the x coordinate. We have 20 classes representing a division of the
# x space into 20 equally wide regions.
labels_df["y_class"] = (np.floor(labels_df["y"] / 0.05)).astype(int)
# Drop original labels
labels_df = labels_df.drop(columns=["x", "y"])

# Get the rows corresponding to training and validation sets.
val_labels_df = labels_df[labels_df["file_name"].isin(VALIDATION_NAMES)]
train_labels_df = labels_df[~labels_df["file_name"].isin(VALIDATION_NAMES)]
# Store the label names separately
val_labels_df.to_csv("../data/labels/multiclass_validation_labels.txt", sep=" ", index=False, header=False)
train_labels_df.to_csv("../data/labels/multiclass_train_labels.txt", sep=" ", index=False, header=False)