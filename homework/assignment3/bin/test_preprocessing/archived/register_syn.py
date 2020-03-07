# Register all training volumes to 0007.nii.gz. No resizing in this version

import ants
import os

# Constants for path names
FIXED_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/0007.nii.gz"
OLD_TEST_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img/"
NEW_TEST_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img_registered_syn/"
fixed = ants.image_read(FIXED_IMG)
fixed = ants.resample_image(fixed, [256, 256, 80], True, 1)

# Register all the training images
for file_name in os.listdir(OLD_TEST_IMG):
    moving_image = ants.image_read(OLD_TEST_IMG + file_name)
    # Downsample for faster registration
    moving_image = ants.resample_image(moving_image, [256, 256, 80], True, 1)
    print("Registering ", file_name)
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform='SyN' )
    print(transform)
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    # Upsample again
    transformed_image = ants.resample_image(transformed_image, [512, 512, 160], True, 1)
    transformed_image.to_file(NEW_TEST_IMG + file_name)
    print("Saving ", file_name)
