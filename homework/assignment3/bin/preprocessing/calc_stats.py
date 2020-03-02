import nibabel as nib
import numpy as np
import os

ORIGINAL_LABELS = "../../data/Original_Training/label/"

# Print out statistics for original data
print("######################################################")
print("Original Statistics")
print("######################################################")
for file_name in os.listdir(ORIGINAL_LABELS):
    # Load the image
    image = nib.load(ORIGINAL_LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    print(file_name + " shape of Data: ", image_data.shape)

    # Filter by 1 values for spleen
    spleen_labels = np.where(image_data == 1, 1, 0)
    # Find where spleen starts and ends on z axis
    spleen_z = np.sum(spleen_labels, axis=(0, 1))
    spleen_z = np.where(spleen_z > 0, 1, 0)
    spleen_z_sum = np.sum(spleen_z)
    print("Percentage of slices on z axis with spleen: ", spleen_z_sum / spleen_labels.shape[2])

    # Calculate the smallest and largest index with spleen label and normalize to
    # 0-1 range since we have variable number of z slices
    spleen_indices = np.nonzero(spleen_z)
    min_spleen = np.min(spleen_indices[0])
    max_spleen = np.max(spleen_indices[0])
    print("Normalized spleen start: ", min_spleen / spleen_labels.shape[2])
    print("Normalized spleen end: ", max_spleen / spleen_labels.shape[2])
    print("Slices before spleen: ", min_spleen )
    print("Slices after spleen: ", spleen_labels.shape[2] - max_spleen)