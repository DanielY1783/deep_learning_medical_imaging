from matplotlib import pyplot as plt
import numpy as np
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

loss = nn.L1Loss(reduction="sum")
loss2 = nn.L1Loss(reduction="none")
input = torch.randn(3, 5)
target = torch.randn(3, 5)
output = loss(input, target)
output2 = loss2(input, target)
print(input)
print(target)
print(output)
print(output2)
# print(input.size())
# print(target.size())
# print(output.size())
# print(output2.size())