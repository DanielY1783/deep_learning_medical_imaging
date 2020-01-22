# Imports
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Constants
MODEL_NAME = "network1.pt"

# Define the neural network
class Net(nn.Module):
    # Define the dimensions for each layer.
    def __init__(self):
        super(Net, self).__init__()
        # First two convolutional layers
        self.conv1 = nn.Conv2d(3, 15, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(15)
        self.conv2 = nn.Conv2d(15, 15, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(15)


        # Two more convolutional layers before maxpooling
        self.conv3 = nn.Conv2d(15, 30, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(30)
        self.conv4 = nn.Conv2d(30, 30, 3, 1)
        self.conv4_bn = nn.BatchNorm2d(30)

        # Two more convolutional layers before maxpooling
        self.conv5 = nn.Conv2d(30, 60, 3, 1)
        self.conv5_bn = nn.BatchNorm2d(60)
        self.conv6 = nn.Conv2d(60, 60, 3, 1)
        self.conv6_bn = nn.BatchNorm2d(60)

        # Two more convolutional layers before maxpooling
        self.conv7 = nn.Conv2d(60, 120, 3, 1)
        self.conv7_bn = nn.BatchNorm2d(120)
        self.conv8 = nn.Conv2d(120, 120, 3, 1)
        self.conv8_bn = nn.BatchNorm2d(120)

        # Dropout values for convolutional and fully connected layers
        self.dropout1 = nn.Dropout2d(0.01)
        self.dropout2 = nn.Dropout2d(0.01)

        # Two fully connected layers. Input is 2347380 because 243x161x60
        # as shown in the forward part.
        self.fc1 = nn.Linear(55080, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 4)

    # Define the structure for forward propagation.
    def forward(self, x):
        # Input dimensions: 490x326x3
        # Output dimensions: 488x324x30
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 488x324x30
        # Output dimensions: 486x322x30
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 486x322x30
        # Output dimensions: 243x161x30
        x = F.max_pool2d(x, 2)

        # Input dimensions: 243x161x30
        # Output dimensions: 241x159x60
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 241x159x60
        # Output dimensions: 239x157x60
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 239x157x60
        # Output dimensions: 120x79x60
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 120x79x60
        # Output dimensions: 118x77x120
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 118x77x120
        # Output dimensions: 116x75x120
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 116x75x120
        # Output dimensions: 58x38x120
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 58x38x120
        # Output dimensions: 56x36x240
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 56x36x240
        # Output dimensions: 54x34x240
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 54x34x240
        # Output dimensions: 27x17x240
        x = F.max_pool2d(x, 2, ceil_mode=True)


        # Input dimensions: 27x17x240
        # Output dimensions: 110160x1
        x = torch.flatten(x, 1)
        # Input dimensions: 110160x1
        # Output dimensions: 128x1
        x = self.fc1(x)
        x = self.fc1_bn(x)
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
        # # Print output with prettier formatting
        print(round(output[:, 0].item(), 4), round(output[:, 1].item(), 4),
              round(output[:, 2].item(), 4), round(output[:, 3].item(), 4))


if __name__ == '__main__':
    main()