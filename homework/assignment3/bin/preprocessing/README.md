Preprocessing Pipeline

1. Train/Validation Split with train_val_split.py
2. Register all volumes to common space with register.py. Volumes are resized to 224x224x70 first for quicker registration. Note that 2 and 4 are thrown out due to registration failure.
3. Rescale all images to -1 to 1 range