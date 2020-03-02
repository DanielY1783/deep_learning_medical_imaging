# Register all volumes to 0001.nii.gz
import ants

# Constants for path names
# FIXED_IMG = "../../data/Train/img/0001.nii.gz"
# # OLD_TRAIN_IMG = "../../data/Train/img_rescale/"
# # NEW_TRAIN_IMG = "../../data/Train/img_registered/"
# # OLD_VAL_LABELS_FILTERED = "../../data/Val/label_filtered/"
# # NEW_VAL_LABELS_FILTERED = "../../data/Val/label_filtered_2d/"
# # OLD_VAL_LABELS = "../../data/Val/label/"
# # NEW_VAL_LABELS = "../../data/Val/label_2d/"
# # OLD_VAL_IMG = "../../data/Val/img/"
# # NEW_VAL_IMG = "../../data/Val/img_2d/"
# #
# # fixed = ants.image_read(FIXED_IMG)
# # for file_name in os.listdir(OLD_TRAIN_LABELS_FILTERED):
# # moving = ants.image_read( ants.get_ants_data('r64') )
# # fixed = ants.resample_image(fixed, (64,64), 1, 0)
# # moving = ants.resample_image(moving, (64,64), 1, 0)
# # mytx = ants.registration(fixed=fixed , moving=moving ,
# #                              type_of_transform = 'SyN' )
# # mywarpedimage = ants.apply_transforms( fixed=fixed, moving=moving,
# #                                            transformlist=mytx['fwdtransforms'] )