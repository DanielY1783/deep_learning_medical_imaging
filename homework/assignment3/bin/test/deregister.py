# Register the labels back to original space

import ants
import os
# Constants for path names
VAL_IMG = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img/"
VAL_IMG_REGISTER = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img_registered_all/"
VAL_PREDICTIONS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/results/Testing/predictions_thresholded/"
NEW_VAL_PREDICTIONS = "/content/drive/My Drive/cs8395_deep_learning/assignment3/results/Testing/predictions_final/"

# Transform all validation predictions back to original space
for file_name in os.listdir(VAL_IMG):
    fixed = ants.image_read(VAL_IMG + file_name)
    moving_image = ants.image_read(VAL_IMG_REGISTER + file_name)
    label = ants.image_read(VAL_PREDICTIONS + file_name)
    # Affine register, except for 0066.nii.gz, which was registered with rigid
    if file_name != "0066.nii.gz":
        transform = ants.registration(fixed=fixed, moving=moving_image,
                                     type_of_transform = 'Affine', random_seed=0)
    else:
        transform = ants.registration(fixed=fixed, moving=moving_image,
                                      type_of_transform = 'Rigid', random_seed=0)
    # Apply transform back to original space
    deregistered_label = ants.apply_transforms(fixed=fixed, moving=label,
                                               transformlist=transform['fwdtransforms'],
                                               interpolator='nearestNeighbor')
    deregistered_label.to_file(NEW_VAL_PREDICTIONS + file_name)