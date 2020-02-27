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

image = np.load("../../data/Training_2d/img/0001_130.npy")
image_tensor = torch.from_numpy(image).float()
image_tensor = torch.unsqueeze(image_tensor, 0)
print(image_tensor.element_size() * image_tensor.nelement())

image2 = np.array(Image.open("../../../assignment2/data/resized224/train/ISIC_0024306.jpg"))
image_tensor2 = torch.from_numpy(image2).float()
image_tensor2 = torch.unsqueeze(image_tensor2, 0)
print(image_tensor2.element_size() * image_tensor2.nelement())