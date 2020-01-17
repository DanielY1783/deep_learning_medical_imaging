# Imports
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Constants
MODEL_NAME = "architecture1.pt"

# Define the neural network
class Net(nn.Module):
    # Define the dimensions for each layer.
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer has 3 input channels, 15 output channels,
        # a 3x3 square kernel, and a stride of 1.
        self.conv1 = nn.Conv2d(3, 15, 3, 1)
        # Second convolutional layer has 30 input channels, 30 output channels,
        # a 3x3 square kernel, and a stride of 1.
        self.conv2 = nn.Conv2d(15, 30, 3, 1)
        # Dropout is performed twice in the network,
        # with the first time set to 0.25 and the
        # second time set to 0.5.
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # Two fully connected layers. Input is 2347380 because 243x161x60
        # as shown in the forward part.
        self.fc1 = nn.Linear(290400, 128)
        # Second fully connected layer has 128 inputs and 2 outputs for x and y values
        self.fc2 = nn.Linear(128, 2)

    # Define the structure for forward propagation.
    def forward(self, x):
        # Input dimensions: 490x326x3
        # Output dimensions: 488x324x15
        x = self.conv1(x)
        # Input dimensions: 488x324x15
        # Output dimensions: 244x162x15
        x = F.max_pool2d(x, 2)
        # Input dimensions: 244x162x15
        # Output dimensions: 242x160x15
        x = self.conv2(x)
        # Input dimensions: 242x160x30
        # Output dimensions: 121x80x30
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        # Input dimensions: 121x80x30
        # Output dimensions: 290400x1
        x = torch.flatten(x, 1)
        # Input dimensions: 290400x1
        # Output dimensions: 128x1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        # Input dimensions: 128x1
        # Output dimensions: 2x1
        output = self.fc2(x)
        return output

def main():
    # Command line arguments for the image path and x and y coordinates
    parser = argparse.ArgumentParser(description='Visualize a Single Prediction Location')
    parser.add_argument('image_path', help='path to the image to display')
    args = parser.parse_args()

    # Open the image passed by the command line argument
    image = Image.open(args.image_path)
    # Convert to numpy array and transpose to get right dimensions
    image = np.array(image)
    image = image.transpose((2, 0, 1))
    # Convert to torch image
    image = torch.from_numpy(image).float()
    # Normalize image
    in_transform = transforms.Compose(
        [transforms.Normalize([146.5899, 142.5595, 139.0785], [34.5019, 34.8481, 37.1137])])
    image = in_transform(image)
    # unsqueeze to insert first dimension for number of images
    image = torch.unsqueeze(image, 0)

    # Specify cuda device
    device = torch.device("cuda")

    # Load in pytorch model
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_NAME))

    # Send image to cuda device
    image = image.to(device, dtype=torch.float32)

    # Specify that we are in evaluation phase
    model.eval()
    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        output = model(image)
        # Print output with prettier formatting
        print(round(output[:, 0].item(), 4), round(output[:, 1].item(), 4))


if __name__ == '__main__':
    main()