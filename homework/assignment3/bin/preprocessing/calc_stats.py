import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

LABELS = "../../data/Val/label_registered/"

# Print out statistics for original data
print("######################################################")
print("Statistics For Images at: ", LABELS)
print("######################################################")
# Lists for start and end indices of spleen for each image
min_list = []
max_list = []
for file_name in os.listdir(LABELS):
    # Load the image
    image = nib.load(LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()
    print(file_name + " shape of Data: ", image_data.shape)

    # Filter by 1 values for spleen
    spleen_labels = np.where(image_data == 1, 1, 0)
    # Find where spleen starts and ends on z axis
    spleen_z = np.sum(spleen_labels, axis=(0, 1))
    spleen_z = np.where(spleen_z > 0, 1, 0)
    spleen_z_sum = np.sum(spleen_z)
    # Calculate the smallest and largest index with spleen label
    spleen_indices = np.nonzero(spleen_z)
    if spleen_z_sum > 0:
        min_spleen = np.min(spleen_indices[0])
        max_spleen = np.max(spleen_indices[0])
        print("Spleen start: ", min_spleen)
        print("Spleen end: ", max_spleen)
        min_list.append(min_spleen)
        max_list.append(max_spleen)

# Print out mean and standard deviation values for start and end of spleen in slices.
print("Mean start slice for spleen: ", np.mean(np.array(min_list)))
print("Std start slice for spleen: ", np.std(np.array(min_list)))
print("Smallest start slice for spleen", np.min(np.array(min_list)))
print("Largest start slice for spleen", np.max(np.array(min_list)))
print("Mean end slice for spleen: ", np.mean(np.array(max_list)))
print("Std end slice for spleen: ", np.std(np.array(max_list)))
print("Smallest end slice for spleen", np.min(np.array(max_list)))
print("Largest end slice for spleen", np.max(np.array(max_list)))
# Plot the start and end slice distributions
plt.hist(np.array(min_list), bins=70)
plt.show()
plt.close()
plt.hist(np.array(max_list), bins=70)
plt.show()
plt.close()