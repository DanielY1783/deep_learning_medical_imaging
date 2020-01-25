# Imports
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Constants
MODEL_NAME = "network.pt"

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
        self.dropout1 = nn.Dropout2d(0.45)
        self.dropout2 = nn.Dropout2d(0.45)

        # Two fully connected layers. Input is 55080 because the last maxpool layer before is
        # 27x17x120 as shown in the forward part.
        self.fc1x = nn.Linear(55080, 256)
        self.fc1x_bn = nn.BatchNorm1d(256)
        self.fc1y = nn.Linear(55080, 256)
        self.fc1y_bn = nn.BatchNorm1d(256)
        # 20 different output nodes for each of the classes, because we divide both
        # the x and y space into 20 spaces. We need two for x and y labels
        self.fc2x = nn.Linear(256, 20)
        self.fc2y = nn.Linear(256, 20)

    # Define the structure for forward propagation.
    def forward(self, x):
        # Input dimensions: 490x326x3
        # Output dimensions: 488x324x15
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 488x324x15
        # Output dimensions: 486x322x15
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 486x322x15
        # Output dimensions: 243x161x15
        x = F.max_pool2d(x, 2)

        # Input dimensions: 243x161x15
        # Output dimensions: 241x159x30
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 241x159x30
        # Output dimensions: 239x157x30
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 239x157x30
        # Output dimensions: 120x79x30
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 120x79x30
        # Output dimensions: 118x77x60
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 118x77x60
        # Output dimensions: 116x75x60
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 116x75x60
        # Output dimensions: 58x38x60
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 58x38x60
        # Output dimensions: 56x36x120
        x = self.conv7(x)
        x = self.conv7_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 56x36x120
        # Output dimensions: 54x34x120
        x = self.conv8(x)
        x = self.conv8_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 54x34x120
        # Output dimensions: 27x17x120
        x = F.max_pool2d(x, 2, ceil_mode=True)


        # Input dimensions: 27x17x120
        # Output dimensions: 55080x1
        x = torch.flatten(x, 1)

        # Fully connected layers for x label prediction
        # Input dimensions: 55080x1
        # Output dimensions: 256x1
        x_label = self.fc1x(x)
        x_label = self.fc1x_bn(x_label)
        x_label = F.relu(x_label)
        x_label = self.dropout2(x_label)
        # Input dimensions: 256x1
        # Output dimensions: 20x1
        x_label = self.fc2x(x_label)

        # Fully connected layers for y label prediction
        # Input dimensions: 55080x1
        # Output dimensions: 256x1
        y_label = self.fc1y(x)
        y_label = self.fc1y_bn(y_label)
        y_label = F.relu(y_label)
        y_label = self.dropout2(y_label)
        # Input dimensions: 256x1
        # Output dimensions: 20x1
        y_label = self.fc2y(y_label)


        # Use log softmax to get probabilities for each class. We
        # can then get the class prediction by simply taking the index
        # with the maximum value.
        output_x = F.log_softmax(x_label, dim=1)
        output_y = F.log_softmax(y_label, dim=1)
        return output_x, output_y

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

    # Load in pytorch model for prediction
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_NAME))
    # Specify that we are in evaluation phase
    model.eval()

    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # Get the prediction label for x and y
        output_x, output_y = model(image)
        label_x = output_x.argmax(dim=1, keepdim=True)
        label_y = output_y.argmax(dim=1, keepdim=True)

        # Convert to x and y values for center of that label
        pred_x = (label_x * 0.05 + (label_x + 1) * 0.05) / 2
        pred_y = (label_y * 0.05 + (label_y + 1) * 0.05) / 2
        # Calculate the center of the box for that label and print output
        print(round(pred_x.item(), 4), round(pred_y.item(), 4))


if __name__ == '__main__':
    main()