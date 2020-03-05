Preprocessing Pipeline

1. Train/Validation Split with train_val_split.py
2. Register all volumes to common space with register_no_resize.py. Note that 2 and 4 are thrown out due to registration failure.
3. Rescale all images to -1 to 1 range with rescale.py
4. Use calc_stats.py to find where the spleen labels are in on the z-axis
5. Use 2d_slice.py to create 224x224 slices for 2d classification. Only use a portion of the slices with the most spleen labels found using calc_stats.py


TO TRY
SYN registration
Binary Cross Entropy with thresholding
Pretrained 2D Unet
3D Unet