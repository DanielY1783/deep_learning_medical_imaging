# Threshold at 0.01, which was the threshold used for validation set.

import nibabel as nib
import numpy as np
import os

# Path for the predicted volumes
PREDICTION_PATH = "../../results/Testing/prediction_float/"
# Path to save the thresholded volumes
THRESHOLD_PATH = "../../results/Testing/predictions_thresholded/"
# Threshold for prediction of 1 value
THRESHOLD = 0.01

def main():
    # Iterate through all validation volumes and calculate results at different thresholds for each one
    for file_name in os.listdir(PREDICTION_PATH):
        # Load the prediction
        image = nib.load(PREDICTION_PATH + file_name)
        # Get the array of values
        image_data = image.get_fdata()
        # Threshold for predictions
        prediction = np.where(image_data >= THRESHOLD, 1, 0)
        prediction = nib.Nifti1Image(prediction, image.affine)
        # Save the prediction
        nib.save(prediction, THRESHOLD_PATH + file_name)

if __name__ == '__main__':
    main()

