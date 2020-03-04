import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt

LABELS = "../../data/Val/label_registered_no_resize/"

# Print out statistics for original data
print("######################################################")
print("Statistics For Images at: ", LABELS)
print("######################################################")
# Lists for start and end indices of spleen for each image on each axis
min_list_z = []
max_list_z = []
min_list_x = []
max_list_x = []
min_list_y = []
max_list_y = []
for file_name in os.listdir(LABELS):
    # Load the image
    image = nib.load(LABELS + file_name)
    # Get the array of values
    image_data = image.get_fdata()

    # Filter by 1 values for spleen
    spleen_labels = np.where(image_data == 1, 1, 0)

    # Print out the number of spleen labels
    print(file_name, ": ", np.sum(spleen_labels), " total spleen labels")

    # Find where spleen starts and ends on z axis
    spleen_z = np.sum(spleen_labels, axis=(0, 1))
    spleen_z = np.where(spleen_z > 0, 1, 0)
    spleen_z_sum = np.sum(spleen_z)
    # Calculate the smallest and largest index with spleen label
    spleen_indices = np.nonzero(spleen_z)
    if spleen_z_sum > 0:
        min_spleen = np.min(spleen_indices[0])
        max_spleen = np.max(spleen_indices[0])
        min_list_z.append(min_spleen)
        max_list_z.append(max_spleen)

    # Find where spleen starts and ends on x axis
    spleen_x = np.sum(spleen_labels, axis=(1, 2))
    spleen_x = np.where(spleen_x > 0, 1, 0)
    spleen_x_sum = np.sum(spleen_x)
    # Calculate the smallest and largest index with spleen label
    spleen_indices = np.nonzero(spleen_x)
    if spleen_x_sum > 0:
        min_spleen = np.min(spleen_indices[0])
        max_spleen = np.max(spleen_indices[0])
        min_list_x.append(min_spleen)
        max_list_x.append(max_spleen)

    # Find where spleen starts and ends on y axis
    spleen_y = np.sum(spleen_labels, axis=(0, 2))
    spleen_y = np.where(spleen_y > 0, 1, 0)
    spleen_y_sum = np.sum(spleen_y)
    # Calculate the smallest and largest index with spleen label
    spleen_indices = np.nonzero(spleen_y)
    if spleen_x_sum > 0:
        min_spleen = np.min(spleen_indices[0])
        max_spleen = np.max(spleen_indices[0])
        min_list_y.append(min_spleen)
        max_list_y.append(max_spleen)

# Print out mean and standard deviation values for start and end of spleen in slices.
print("Mean z start slice for spleen: ", np.mean(np.array(min_list_z)))
print("Std z start slice for spleen: ", np.std(np.array(min_list_z)))
print("Smallest z start slice for spleen", np.min(np.array(min_list_z)))
print("Largest z start slice for spleen", np.max(np.array(min_list_z)))
print("Mean z end slice for spleen: ", np.mean(np.array(max_list_z)))
print("Std z end slice for spleen: ", np.std(np.array(max_list_z)))
print("Smallest z end slice for spleen", np.min(np.array(max_list_z)))
print("Largest z end slice for spleen", np.max(np.array(max_list_z)))

print("Mean x start slice for spleen: ", np.mean(np.array(min_list_x)))
print("Std x start slice for spleen: ", np.std(np.array(min_list_x)))
print("Smallest x start slice for spleen", np.min(np.array(min_list_x)))
print("Largest x start slice for spleen", np.max(np.array(min_list_x)))
print("Mean x end slice for spleen: ", np.mean(np.array(max_list_x)))
print("Std x end slice for spleen: ", np.std(np.array(max_list_x)))
print("Smallest x end slice for spleen", np.min(np.array(max_list_x)))
print("Largest x end slice for spleen", np.max(np.array(max_list_x)))

print("Mean y start slice for spleen: ", np.mean(np.array(min_list_y)))
print("Std y start slice for spleen: ", np.std(np.array(min_list_y)))
print("Smallest y start slice for spleen", np.min(np.array(min_list_y)))
print("Largest y start slice for spleen", np.max(np.array(min_list_y)))
print("Mean y end slice for spleen: ", np.mean(np.array(max_list_y)))
print("Std y end slice for spleen: ", np.std(np.array(max_list_y)))
print("Smallest y end slice for spleen", np.min(np.array(max_list_y)))
print("Largest y end slice for spleen", np.max(np.array(max_list_y)))
# Plot the start and end slice distributions for z
plt.hist(np.array(min_list_z), bins=70)
plt.show()
plt.close()
plt.hist(np.array(max_list_z), bins=70)
plt.show()
plt.close()
# Plot the start and end slice distributions for x
plt.hist(np.array(min_list_x), bins=70)
plt.show()
plt.close()
plt.hist(np.array(max_list_x), bins=70)
plt.show()
plt.close()
# Plot the start and end slice distributions for y
plt.hist(np.array(min_list_y), bins=70)
plt.show()
plt.close()
plt.hist(np.array(max_list_y), bins=70)
plt.show()
plt.close()