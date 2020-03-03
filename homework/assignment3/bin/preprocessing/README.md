Preprocessing Pipeline

1. Train/Validation Split with train_val_split.py
2. Resize all volumes to 224x224x70 with resize.py
3. Register all volumes to common space with register.py
4. Rescale all values to -1 to 1 range with rescale.py