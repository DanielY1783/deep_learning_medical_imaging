# Imports
import pandas as pd

# Function to generate numerical label from one-hot
def generate_numerical_label(row):
    if row["MEL"] == 1:
        return 0
    elif row["NV"] == 1:
        return 1
    elif row["BCC"] == 1:
        return 2
    elif row["AKIEC"] == 1:
        return 3
    elif row["BKL"] == 1:
        return 4
    elif row["DF"] == 1:
        return 5
    else:
        return 6

# Function to subtract 1 from labels 2-6 for the set without class 1
def relabel_no_class_1(row):
    if row["label"] == 0:
        return 0
    else:
        return row["label"] - 1

# Function to subtract 1 from labels 2-6 for the set without class 1
def binary_1(row):
    if row["label"] == 1:
        return 1
    else:
        return 0

# Load in labels files.
train_labels_df = pd.read_csv("../data/labels/Train_labels.csv", sep=",")
test_labels_df = pd.read_csv("../data/labels/Test_labels.csv", sep=",")

# Append a .jpg for the file name
train_labels_df["image"] = train_labels_df["image"] + ".jpg"
test_labels_df["image"] = test_labels_df["image"] + ".jpg"

# Add new column for integer value for the label
train_labels_df["label"] = train_labels_df.apply(generate_numerical_label, axis=1)
test_labels_df["label"] = test_labels_df.apply(generate_numerical_label, axis=1)

# Get total number of training and testing instances
num_train_images = train_labels_df.shape[0]
num_test_images = test_labels_df.shape[0]
# Print out fraction of images for each label
for label in ["MEL","NV","BCC","AKIEC","BKL","DF","VASC"]:
    # Get number of instances with that label
    train_instances = train_labels_df[label].sum()
    test_instances = test_labels_df[label].sum()
    # Print out fraction of instances
    print("Percentage of label ", label)
    print("Training: ", float(train_instances/num_train_images))
    print("Testing: ", float(test_instances/num_test_images))

# Drop original one-hot label columns
train_labels_df = train_labels_df.drop(columns=["MEL","NV","BCC","AKIEC","BKL","DF","VASC"])
test_labels_df = test_labels_df.drop(columns=["MEL","NV","BCC","AKIEC","BKL","DF","VASC"])
# Store the label names
train_labels_df.to_csv("../data/labels/formatted_train_labels.csv", sep="\t", index=False, header=False)
test_labels_df.to_csv("../data/labels/formatted_test_labels.csv", sep="\t", index=False, header=False)
