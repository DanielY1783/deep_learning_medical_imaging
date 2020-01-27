# Name: Daniel Yan
# Email: daniel.yan@vanderbilt.edu
# Description: Quick script to calculate the mean and standard deviation for each channel of the
# training images to normalize them before passing to neural network.

# Imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from skimage import io, transform


# Class for the dataset
class DetectionImages(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file, sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_df.iloc[idx, 0])
        image = io.imread(img_name)
        label = self.labels_df.iloc[idx, 1:]
        label = np.array([label])
        label = label.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


# Load in the training and testing datasets. Convert to pytorch tensor.
train_data = DetectionImages(csv_file="../data/labels/train_labels.txt", root_dir="../data/train")
train_loader = DataLoader(train_data, batch_size=1000, shuffle=True, num_workers=0)

# Get just the images
image_array = None
for index, images in enumerate(train_loader):
    image_array = images["image"]

image_array = image_array.float()

# Get the red, blue, green channels
red = image_array[:, :, :, 0]
blue = image_array[:, :, :, 1]
green = image_array[:, :, :, 2]

print("Red Mean: ", red.mean())
print("Blue Mean: ", blue.mean())
print("Green Mean: ", green.mean())

print("Red Std: ", red.std())
print("Blue Std: ", blue.std())
print("Green Std: ", green.std())