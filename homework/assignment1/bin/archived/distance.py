# Imports
import numpy as np
import os
import pandas as pd
from PIL import Image
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Constant for names of validation files
VALIDATION_NAMES = ["111.jpg", "112.jpg", "113.jpg", "114.jpg", "115.jpg",
                    "116.jpg", "117.jpg", "118.jpg", "119.jpg", "125.jpg"]
# Network name to load
MODEL_NAME = "network.pt"
# Number of windows in both x and y coordinates, which is the number of labels
WINDOWS = 20

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
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Normalize images with mean and standard deviation from each channel found using some
        # simple array calculations
        in_transform = transforms.Compose([transforms.Normalize([146.5899, 142.5595, 139.0785], [34.5019, 34.8481, 37.1137])])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image = in_transform(image)
        return {'image': image,
                'label': torch.from_numpy(np.array(label).astype(int))}

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
    # Load in labels file and rename column names.
    labels_df = pd.read_csv("../data/labels/labels.txt", sep=" ", header=None)
    labels_df.columns = ["file_name", "x", "y"]

    # Get the rows corresponding to validation set.
    val_labels_df = labels_df[labels_df["file_name"].isin(VALIDATION_NAMES)]
    # Drop file names
    val_labels_df = val_labels_df .drop(columns=["file_name"])
    # Convert to numpy array
    val_labels_np = np.array(val_labels_df)
    # Get the x and y values in separately arrays
    val_labels_x = val_labels_np[:, 0]
    val_labels_y = val_labels_np[:, 1]

    # Load in the images
    test_data = DetectionImages(csv_file="../data/labels/validation_labels.txt", root_dir="../data/validation", transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=len(VALIDATION_NAMES), shuffle=False, num_workers=0)

    # Specify cuda device
    device = torch.device("cuda")

    # Load in pytorch model for prediction
    model = Net().to(device)
    model.load_state_dict(torch.load(MODEL_NAME))
    # Specify that we are in evaluation phase
    model.eval()

    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(test_loader):
            # Send training data and the training labels to GPU/CPU
            data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device,
                                                                                                           dtype=torch.long)
            # Obtain the output from the model
            output_x, output_y = model(data)
            # Convert softmax output to label
            output_x = output_x.argmax(dim=1, keepdim=True)
            output_y = output_y.argmax(dim=1, keepdim=True)
            # Convert outputs to numpy arrays. Slice to change to 1d numpy array
            output_x_np, output_y_np = output_x.cpu().numpy()[:,0], output_y.cpu().numpy()[:,0]

            # Convert the predicted labels to floating point values corresponding to the middle of that
            # "window"
            pred_x = (output_x_np / WINDOWS + (output_x_np + 1) / WINDOWS) / 2
            pred_y = (output_y_np / WINDOWS + (output_y_np + 1) / WINDOWS) / 2

            # Calculate the euclidean distance from prediction to actual floating point value
            distance_squared = np.square(val_labels_x-pred_x) + np.square(val_labels_y-pred_y)
            distance = np.sqrt(distance_squared)

            # Print the distance for each prediction and the average distance
            print("Distance For Each Validation Set Prediction: ", distance)
            print("Average Distance: ", np.mean(distance))

if __name__ == '__main__':
    main()