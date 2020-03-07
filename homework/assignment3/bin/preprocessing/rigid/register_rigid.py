# Register all training volumes to 0007.nii.gz. No resizing in this version

import ants
import os

# Constants for path names
FIXED_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/0007.nii.gz"
OLD_TRAIN_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/"
NEW_TRAIN_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/rigid/img_register_rigid/"
OLD_TRAIN_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/label/"
NEW_TRAIN_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/rigid/label_register_rigid/"
OLD_VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/img/"
NEW_VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/rigid/img_register_rigid/"
OLD_VAL_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/label/"
NEW_VAL_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/rigid/label_register_rigid/"

fixed = ants.image_read(FIXED_IMG)

# Register all the training images
for file_name in os.listdir(OLD_TRAIN_IMG):
    moving_image = ants.image_read(OLD_TRAIN_IMG + file_name)
    label = ants.image_read(OLD_TRAIN_LABELS + file_name)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform='QuickRigid', random_seed=0)
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
    label = ants.image_read(OLD_VAL_LABELS + file_name)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform = 'QuickRigid', random_seed=0)
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator = 'nearestNeighbor')
    transformed_image.to_file(NEW_VAL_IMG + file_name)
    transformed_label = ants.apply_transforms( fixed=fixed, moving=label,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator = 'nearestNeighbor')
    transformed_label.to_file(NEW_VAL_LABELS + file_name)