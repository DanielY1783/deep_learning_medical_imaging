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
        # Create list of images
        images_list = []
        for file_name in os.listdir(image_path):
            # Load in image using numpy
            image = np.load(image_path + file_name)
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image).float()
            # Insert first dimension for number of channels
            image_tensor = torch.unsqueeze(image_tensor, 0)
            # Add to list of images.
            images_list.append(image_tensor)
        self.images_list = images_list
        # Create list of target segmentations
        targets_list = []
        for file_name in os.listdir(target_path):
            mask = np.load(target_path + file_name)
            # Convert to torch tensor
            mask_tensor = torch.from_numpy(mask).float()
            # Insert first dimension for number of channels
            mask_tensor = torch.unsqueeze(mask_tensor, 0)
            # Add to list of masks.
            targets_list.append(mask_tensor)
        self.targets_list = targets_list


    def __getitem__(self, index):
        return self.images_list[index], self.targets_list[index]


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
    train_data = MyDataset(image_path = "../../data/Train/img_2d/", target_path = "../../data/Train/label_2d/")
    val_data = MyDataset(image_path = "../../data/Val/img_2d/", target_path = "../../data/Val/label_2d/")
    # Create data loader for training and validation
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # for i in range(len(train_data.targets_list)):
    #     if i % 100 == 0:
    #         print(train_data.targets_list[i].shape)
    #         print(train_data.images_list[i].shape)
    #         print(train_data.targets_list[i])
    #         print(train_data.images_list[i])



if __name__ == '__main__':
    main()