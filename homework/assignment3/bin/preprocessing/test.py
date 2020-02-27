from PIL import Image
import numpy as np
import nibabel as nib

y = np.zeros((512, 512, 147))
z = np.zeros((512, 512, 147))
for i in range(147):
    x = np.load("../../data/Training_2d/img/0001_" + str(i) + ".npy")
    x2 = np.load("../../data/Training_2d/label/0001_" + str(i) + ".npy")
    y[:, :, i] = x
    z[:, :, i] = x2

image_original = nib.load("../../data/Training/img/img0001.nii.gz")
array_img = nib.Nifti1Image(y, image_original.affine)
nib.save(array_img, '../../data/Training_2d/img/my_image.nii')

image_original2 = nib.load("../../data/Training/label/label0001.nii.gz")
array_img2 = nib.Nifti1Image(z, image_original2.affine)
nib.save(array_img2, '../../data/Training_2d/img/my_seg.nii')

