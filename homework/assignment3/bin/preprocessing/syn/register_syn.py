# Register all training volumes to 0007.nii.gz. No resizing in this version

import ants
import os

# Constants for path names
FIXED_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/0007.nii.gz"
OLD_TRAIN_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/"
NEW_TRAIN_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/syn/img_register/"
OLD_TRAIN_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/label/"
NEW_TRAIN_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/syn/label_register/"
OLD_VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/img/"
NEW_VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/syn/img_register/"
OLD_VAL_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/label/"
NEW_VAL_LABELS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/syn/label_register/"

fixed = ants.image_read(FIXED_IMG)
fixed = ants.resample_image(fixed, [256, 256, 80], True, 1)

# Register all the training images
for file_name in os.listdir(OLD_TRAIN_IMG):
    moving_image = ants.image_read(OLD_TRAIN_IMG + file_name)
    label = ants.image_read(OLD_TRAIN_LABELS + file_name)
    # Downsample for faster registration
    moving_image = ants.resample_image(moving_image, [256, 256, 80], True, 1)
    label = ants.resample_image(label, [256, 256, 80], True, 1)
    print("Registering ", file_name)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform='SyN' )
    print(transform)
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    # Upsample again
    transformed_image = ants.resample_image(transformed_image, [512, 512, 160], True, 1)
    transformed_image.to_file(NEW_TRAIN_IMG + file_name)
    transformed_label = ants.apply_transforms( fixed=fixed, moving=label,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    transformed_label = ants.resample_image(transformed_label, [512, 512, 160], True, 1)
    transformed_label.to_file(NEW_TRAIN_LABELS + file_name)
    print("Saving ", file_name)

# Repeat for the validation images
for file_name in os.listdir(OLD_VAL_IMG):
    moving_image = ants.image_read(OLD_VAL_IMG + file_name)
    label = ants.image_read(OLD_VAL_LABELS + file_name)
    # Downsample for faster registration
    moving_image = ants.resample_image(moving_image, [256, 256, 80], True, 1)
    label = ants.resample_image(label, [256, 256, 80], True, 1)
    print("Registering ", file_name)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform='SyN' )
    print(transform)
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator  = 'nearestNeighbor')
    # Upsample again
    transformed_image = ants.resample_image(transformed_image, [512, 512, 160], True, 1)
    transformed_image.to_file(NEW_VAL_IMG + file_name)
    transformed_label = ants.apply_transforms( fixed=fixed, moving=label,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator  = 'nearestNeighbor')
    transformed_label = ants.resample_image(transformed_label, [512, 512, 160], True, 1)
    transformed_label.to_file(NEW_VAL_LABELS + file_name)
    print("Saving ", file_name)