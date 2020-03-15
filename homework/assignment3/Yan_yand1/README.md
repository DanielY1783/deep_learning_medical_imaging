Descriptions for the source files. Run files in the order they are listed below. The final predictions are in the folder Segmentation. Note that path name
constants at the top of the files may need to be modified to run code.

# Preprocessing
The files below are used to preprocess the training and validation sets.
## train_val_split.py
Split the data into training and validation sets.

## register_affine.py
Affinely register the images to the same space

## register_rigid.py
Rigid registration of images to the same space. Use this for images where affine registration failed. After running
this, replace any failed affine results with the results from the rigid registration.

## calc_stats.py
Calculate where the spleen labels start and end in the different images. Results are used to determine how to create 2D slices.

## 2d_slice.py
Create 2d slices of the 2d volumes to feed into a 2d segmentation network. Normalize the images by dividing by 1000 to get into -1 to 1 range.

# test_preprocessing
The files below repeat the registration for the testing set.

## register_affine.py
Affinely register the images to the same space

## register_rigid.py
Rigid registration of images to the same space. Use this for images where affine registration failed. After running
this, replace any failed affine results with the results from the rigid registration.

# train.py
Train the segmentation network using deeplabv3 torchvision network.

# calc_val_volumes.py
Use the trained network to predict validation volumes. This generates a probability map that needs to be postprocessed.

# calc_test_volumes.py
Use the trained network to predict testing volumes. This generates a probability map that needs to be postprocessed.

# postprocessing
The files below postprocess the validation set to get final validation predictions.

## calc_threshold.py
Calculate the threshold for a prediction be to counted as a "1" and use the threshold with the best F1 score

## deregister.py
Register the volumes back to the original space.

# calc_stats.py
Calculate the stats for the volumes back in the original space.

# test_postprocessing
Postprocess the testing set to get final predicitons for the testing set.

## threshold.py
Threshold the volumes using the threshold from the validation set.

## deregister.py
Register the testing volumes back to original space.