# Imports
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Constants
MODEL_NAME_X = "network2_x.pt"
MODEL_NAME_Y = "network2_y.pt"

# Define the neural network
class Net(nn.Module):
    # Define the dimensions for each layer.
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional Layers with batch normalization
        self.conv1 = nn.Conv2d(3, 10, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(40)

        # Dropout values for convolutional and fully connected layers
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)

        # Two fully connected layers. Input is 2347380 because 243x161x60
        # as shown in the forward part.
        self.fc1 = nn.Linear(5040, 128)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 20)

    # Define the structure for forward propagation.
    def forward(self, x):
        # Input dimensions: 490x326x3
        # Output dimensions: 488x324x10
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 488x324x10
        # Output dimensions: 122x81x10
        x = F.max_pool2d(x, 4)

        # Input dimensions: 122x81x10
        # Output dimensions: 120x79x20
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 120x79x20
        # Output dimensions: 30x20x20
        x = F.max_pool2d(x, 4, ceil_mode=True)

        # Input dimensions: 30x20x20
        # Output dimensions: 28x18x40
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 28x18x40
        # Output dimensions: 14x9x40
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 14x9x40
        # Output dimensions: 5040x1
        x = torch.flatten(x, 1)
        # Input dimensions: 5040x1
        # Output dimensions: 128x1
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout2(x)
        # Input dimensions: 128x1
        # Output dimensions: 20x1
        x = self.fc2(x)
        output = torch.sigmoid(x)
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


    # Send image to cuda device
    image = image.to(device, dtype=torch.float32)

    # Load in pytorch model for x prediction
    model_x = Net().to(device)
    model_x.load_state_dict(torch.load(MODEL_NAME_X))
    # Specify that we are in evaluation phase
    model_x.eval()

    # Load in pytorch model for y prediction
    model_y = Net().to(device)
    model_y.load_state_dict(torch.load(MODEL_NAME_Y))
    # Specify that we are in evaluation phase
    model_y.eval()
    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # Get the prediction label for x and y
        output_x = model_x(image)
        label_x = output_x.argmax(dim=1, keepdim=True)
        # Convert to x value for center of that label
        pred_x = (label_x * 0.05 + (label_x + 1) * 0.05) / 2

        output_y = model_y(image)
        label_y = output_y.argmax(dim=1, keepdim=True)
        pred_y = (label_y * 0.05 + (label_y + 1) * 0.05) / 2
        # Calculate the center of the box for that label and print output
        print(round(pred_x.item(), 4), round(pred_y.item(), 4))
        print(output_x)
        print(output_y)


if __name__ == '__main__':
    main()