# Calculate F1 score at different thresholds.

import nibabel as nib
import numpy as np
import os

# Path for the predicted volumes
PREDICTION_PATH = "../../results/Val/affine_fixed/prediction_float/"
# Path for the actual labels
LABELS_PATH = "../../data/Val/affine_fixed/label_registered/"
# Path to save the thresholded volumes
THRESHOLD_PATH = "../../results/Val/affine_fixed/prediction_thresholded/"

def main():
    # Step 1: Find the best threshold for counting a prediction as class 1

    # Best threshold and f1 so far
    best_threshold = 0
    best_f1 = 0
    # Iterate through different thresholds to calculate f1 at each threshold
    for threshold in [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]:
        # List of precision, recall, and f1 scores
        precision_list = []
        recall_list = []
        f1_list = []
        # Iterate through all validation volumes and calculate results at different thresholds for each one
        for file_name in os.listdir(PREDICTION_PATH):
            # Load in the actual labels
            label = nib.load(LABELS_PATH + file_name)
            label = label.get_fdata()
            # Filter for only spleen labels
            label = np.where(label == 1, 1, 0)

            # Load the prediction
            image = nib.load(PREDICTION_PATH + file_name)
            # Get the array of values
            image_data = image.get_fdata()
            # Threshold for predictions
            prediction = np.where(image_data >= threshold, 1, 0)

            # Calculate true positives, false positives, and false negatives
            tp = np.where(np.logical_and(label == 1, prediction == 1))
            tp = np.sum(tp)
            fp = np.where(np.logical_and(label == 0, prediction == 1))
            fp = np.sum(fp)
            fn = np.where(np.logical_and(label == 1, prediction == 0))
            fn = np.sum(fn)

            # Calculate precision, recall, and f1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)

            # Add to list of precision, recall, f1
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        # Print precision, recall, f1 at the threshold
        f1 = np.mean(np.array(f1_list))
        precision = np.mean(np.array(precision_list))
        recall = np.mean(np.array(recall_list))
        print("##########################################################################")
        print("Threshold: ", threshold)
        print("##########################################################################")
        print("Precision: ", precision_list)
        print("Precision Mean: ", precision)
        print("Recall: ", recall_list)
        print("Recall Mean: ", recall)
        print("F1: ", f1_list)
        print("F1 Mean: ", f1)

        # Check if this threshold is the best f1 score so far
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold


    # Step 2: Save the predictions at the best threshold
    print("##########################################################################")
    print("Saving the Predictions at the Best Threshold of ", best_threshold)
    print("##########################################################################")
    # Iterate through all validation volumes and calculate results at different thresholds for each one
    for file_name in os.listdir(PREDICTION_PATH):
        # Load the prediction
        image = nib.load(PREDICTION_PATH + file_name)
        # Get the array of values
        image_data = image.get_fdata()
        # Threshold for predictions
        prediction = np.where(image_data >= best_threshold, 1, 0)
        prediction = nib.Nifti1Image(prediction, image.affine)
        # Save the prediction
        nib.save(prediction, THRESHOLD_PATH + file_name)

if __name__ == '__main__':
    main()

