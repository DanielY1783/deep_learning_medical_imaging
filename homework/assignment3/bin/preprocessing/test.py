from PIL import Image
import numpy as np
import nibabel as nib
import os
from sklearn.preprocessing import StandardScaler
OLD_TRAIN_IMG = "../../data/Train/img/"
NEW_TRAIN_IMG = "../../data/Train/img_rescale/"

# y = np.zeros((512, 512, 147))
# z = np.zeros((512, 512, 147))
# for i in range(147):
#     x = np.load("../../data/Training_2d/img/0001_" + str(i) + ".npy")
#     x2 = np.load("../../data/Training_2d/label/0001_" + str(i) + ".npy")
#     y[:, :, i] = x
#     z[:, :, i] = x2
#
# image_original = nib.load("../../data/Training/img/img0001.nii.gz")
# array_img = nib.Nifti1Image(y, image_original.affine)
# nib.save(array_img, '../../data/Training_2d/img/my_image.nii')
#
# image_original2 = nib.load("../../data/Training/label/label0001.nii.gz")
# array_img2 = nib.Nifti1Image(z, image_original2.affine)
# nib.save(array_img2, '../../data/Training_2d/img/my_seg.nii')

# # First for training set
# # Iterate through all the actual images
# for file_name in os.listdir(OLD_TRAIN_IMG):
#     # Create standard scaler
#     scaler = StandardScaler(copy=False)
#     # Load the image
#     image = nib.load(OLD_TRAIN_IMG + file_name)
#     # Get the array of values
#     image_data = image.get_fdata()
#     # # Calculate mean and standard deviation to perform standard scaling
#     # mean = np.mean(image_data)
#     # std = np.std(image_data)
#     # image_data = (image_data - mean) / std
#     # Calculate the minimum value of the data and subtract all values by that value to get min value to 0
#     min = np.amin(image_data)
#     image_data = image_data - min
#     # Calculate the new maximum value and divide by maximum value to get in 0-1 range
#     max = np.amax(image_data)
#     image_data = image_data / max
#     # Subtract 0.45 as mean for imagenet normalization
#     image_data = image_data - 0.45
#     # Divide by 0.225 as standard deviation for imagenet normalization
#     image_data = image_data / 0.225
#     image = nib.Nifti1Image(image_data, image.affine)
#     nib.save(image, NEW_TRAIN_IMG + file_name)

x = np.load("../../data/Val224_2d/img_2d/0004_72.npy")
print(x)

y = np.load("../../data/Val/img_2d/0004_72.npy")
print(y)

z = np.load("../../data/Train224_2d/img_2d/0001_72.npy")
print(z[111, 111])