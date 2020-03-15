# Calculate statistics back in original space
import nibabel as nib
import numpy as np
import os

# Constants for path names
ACTUAL_LABEL = "../../data/Val/label/"
PREDICTED_LABEL = "../../results/Val/affine_fixed/deregistered/"

# List of precision, recall, and f1 scores
precision_list = []
recall_list = []
f1_list = []

# Transform all validation predictions back to original space
for file_name in os.listdir(PREDICTED_LABEL):
    # Load in the actual labels
    actual = nib.load(ACTUAL_LABEL + file_name)
    actual = actual.get_fdata()
    # Filter for only spleen labels
    actual = np.where(actual == 1, 1, 0)

    # Load in predicted labels
    predicted = nib.load(PREDICTED_LABEL + file_name)
    predicted = predicted.get_fdata()

    # Calculate true positives, false positives, and false negatives
    tp = np.where(np.logical_and(actual == 1, predicted == 1))
    tp = np.sum(tp)
    fp = np.where(np.logical_and(actual == 0, predicted == 1))
    fp = np.sum(fp)
    fn = np.where(np.logical_and(actual == 1, predicted == 0))
    fn = np.sum(fn)

    # Calculate precision, recall, and f1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    # Add to list of precision, recall, f1
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

print("Precision: ", precision_list)
print("Recall: ", recall_list)
print("F1: ", f1_list)
print("Mean Precision: ", np.mean(np.array(precision_list)))
print("Mean Recall: ", np.mean(np.array(recall_list)))
print("Mean F1: ", np.mean(np.array(f1_list)))
print("Std F1: ", np.std(np.array(f1_list)))
print("Median F1: ", np.median(np.array(f1_list)))