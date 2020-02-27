# Imports for Pytorch
from __future__ import print_function
import argparse
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from skimage import io, transform

# Define dataset for image and segmentation mask
class MyDataset(Dataset):
    def __init__(self, image_path, target_path):
        # Create list of paths to each image
        images_list = []
        for file_name in os.listdir(image_path):
            images_list.append(image_path + file_name)
        self.images_list = images_list
        # Create list of paths to each target segmentation
        targets_list = []
        for file_name in os.listdir(target_path):
            targets_list.append(target_path + file_name)
        self.targets_list = targets_list

    def __getitem__(self, index):
        # Load in image using numpy
        image = np.load(self.images_list[index])
        mask = np.load(self.targets_list[index])
        # Convert to torch tensor
        image_tensor = torch.from_numpy(image).float()
        # Add first dimension for image having one channel
        image_tensor = torch.unsqueeze(image_tensor, 0)
        # Convert to torch tensor
        mask_tensor = torch.from_numpy(mask).float()
        # Add first dimension for image having one channel
        mask_tensor = torch.unsqueeze(mask_tensor, 0)
        return image_tensor, mask_tensor


    def __len__(self):
        return len(self.images_list)


# Define the UNet Structure

def main():
    # Command line arguments for hyperparameters of model/training.
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--gamma', type=float, default=1, metavar='N',
                        help='gamma value for learning rate decay (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    # Load in the dataset
    data = MyDataset(image_path = "../../data/Training_2d/img/", target_path = "../../data/Training_2d/label/")
    # Split into training and validation
    train_size = int(0.9 * len(data))
    test_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, test_size])
    # Create data loader for training and validation
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)




if __name__ == '__main__':
    main()