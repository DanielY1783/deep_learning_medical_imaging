# Register all testing volumes to 0007.nii.gz. No resizing in this version

import ants
import os

# Constants for path names
FIXED_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/img/0007.nii.gz"
OLD_TEST_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img/"
NEW_TEST_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img_registered_no_resize/"

# Load in fixed image
fixed = ants.image_read(FIXED_IMG)

# Register all the testing images
for file_name in os.listdir(OLD_TEST_IMG):
    # Load in moving image
    moving_image = ants.image_read(OLD_TEST_IMG + file_name)
    print("Registering ", file_name)
    # Perform registration
    transform = ants.registration(fixed=fixed , moving=moving_image,
                                 type_of_transform='AffineFast' )
    transformed_image = ants.apply_transforms( fixed=fixed, moving=moving_image,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    # Save transformed image
    print("Saving ", file_name)
    transformed_image.to_file(NEW_TEST_IMG + file_name)