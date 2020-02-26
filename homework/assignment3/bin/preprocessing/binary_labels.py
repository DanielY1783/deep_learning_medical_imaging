import os
import numpy as np
import nibabel as nib

# Constants
OLD_PATH = "../../data/Training/label/"
NEW_PATH = "../../data/Training/label_filtered/"

# Iterate through all files in labels directory and create binary 0/1 labels for spleen
for file_name in os.listdir(OLD_PATH):
    # Load in nii file
    image = nib.load(OLD_PATH + file_name)
    image_data = image.get_fdata()
    # Filter for 1 label for spleen
    image_data_filtered = np.where(image_data==1, 1.0, 0.0)
    image_filtered = nib.Nifti1Image(image_data_filtered, image.affine)
    # Save the image
    nib.save(image_filtered, NEW_PATH + file_name)