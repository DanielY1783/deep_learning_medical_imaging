# Register all volumes to 0007.nii.gz
# Volumes resized to half size first for quicker registration

import ants
import os

# Constants for path names
FIXED_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/0007.nii.gz"
OLD_TRAIN_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/"
NEW_TRAIN_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img_registeredv2/"
OLD_TRAIN_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/label/"
NEW_TRAIN_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/label_registeredv2/"
OLD_VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/img/"
NEW_VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/img_registeredv2/"
OLD_VAL_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/label/"
NEW_VAL_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/label_registeredv2/"

fixed = ants.image_read(FIXED_IMG)
fixed = ants.resample_image(fixed, [224, 224, int(fixed.numpy().shape[2] * 224/512.0)], True, 1)

# Register all the training images
for file_name in os.listdir(OLD_TRAIN_IMG):
    moving_image = ants.image_read(OLD_TRAIN_IMG + file_name)
    moving_image = ants.resample_image(moving_image, [224, 224, int(moving_image.numpy().shape[2] * 224/512.0)], True, 1)
    label = ants.image_read(OLD_TRAIN_LABELS + file_name)
    label = ants.resample_image(label, [224, 224, int(label.numpy().shape[2] * 224/512.0)], True, 1)
    print("Fixed: ", fixed)
    print("Moving: ", moving_image)
    print("Label: ", label)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform='Affine' )
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    transformed_image.to_file(NEW_TRAIN_IMG + file_name)
    transformed_label = ants.apply_transforms( fixed=fixed, moving=label,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    transformed_label.to_file(NEW_TRAIN_LABELS + file_name)

# Repeat for the validation images
for file_name in os.listdir(OLD_VAL_IMG):
    moving_image = ants.image_read(OLD_VAL_IMG + file_name)
    moving_image = ants.resample_image(moving_image, [224, 224, int(moving_image.numpy().shape[2] * 224/512.0)], True, 1)
    label = ants.image_read(OLD_VAL_LABELS + file_name)
    label = ants.resample_image(label, [224, 224, int(label.numpy().shape[2] * 224/512.0)], True, 1)
    print("Fixed: ", fixed)
    print("Moving: ", moving_image)
    print("Label: ", label)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform = 'Affine' )
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator  = 'nearestNeighbor')
    transformed_image.to_file(NEW_VAL_IMG + file_name)
    transformed_label = ants.apply_transforms( fixed=fixed, moving=label,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator  = 'nearestNeighbor')
    transformed_label.to_file(NEW_VAL_LABELS + file_name)