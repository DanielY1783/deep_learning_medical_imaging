# Imports for Pytorch
from __future__ import print_function
import argparse
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

# Constants
MODEL_NAME = "box_search_l1.pt"

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

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        in_transform = transforms.Compose([transforms.Normalize([146.5899, 142.5595, 139.0785], [34.5019, 34.8481, 37.1137])])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image = in_transform(image)
        label = label.reshape(-1)
        return {'image': image,
                'label': torch.from_numpy(label)}

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
        self.dropout1 = nn.Dropout2d(0.35)
        self.dropout2 = nn.Dropout2d(0.35)

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

def train(args, model, device, train_loader, optimizer, epoch, train_losses):
    # Specify that we are in training phase
    model.train()
    # Total Train Loss
    total_loss = 0
    # Iterate through all minibatches.
    for batch_idx, batch_sample in enumerate(train_loader):
        # Send training data and the training labels to GPU/CPU
        data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device, dtype=torch.float32)
        # Zero the gradients carried over from previous step
        optimizer.zero_grad()
        # Obtain the predictions from forward propagation
        output = model(data)
        # Compute the mean squared error for loss
        l1 = nn.L1Loss()
        loss = l1(output, target)
        total_loss += loss.item()
        # Perform backward propagation to compute the negative gradient, and
        # update the gradients with optimizer.step()
        loss.backward()
        optimizer.step()
    # Update training error and add to accumulation of training loss over time.
    train_error = total_loss / len(train_loader.dataset)
    train_losses.append(train_error)
    # Print output if epoch is finished
    print('Train Epoch: {} \tAverage Loss: {:.6f}'.format(epoch, train_error))
    # Return accumulated losses
    return train_losses


def test(args, model, device, test_loader, test_losses):
    # Specify that we are in evaluation phase
    model.eval()
    # Set the loss initially to 0.
    test_loss = 0
    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # For each testing example, we run forward
        # propagation to calculate the
        # testing prediction. Update the total loss
        # and the number of correct predictions
        # with the counters from above.
        for batch_idx, batch_sample in enumerate(test_loader):
            # Send training data and the training labels to GPU/CPU
            data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device,
                                                                                                           dtype=torch.float32)
            output = model(data)
            l1 = nn.L1Loss()
            test_loss += l1(output, target).item()

    # Average the loss by dividing by the total number of testing instances and add to accumulation of losses.
    test_error = test_loss / len(test_loader.dataset)
    test_losses.append(test_error)

    # Print out the statistics for the testing set.
    print('\nTest set: Average loss: {:.6f}\n'.format(
        test_error))

    # Return accumulated test losses over epochs
    return test_losses, output


def main():
    # Command line arguments for hyperparameters of
    # training and testing batch size, the number of
    # epochs, the learning rate, gamma, and other
    # settings such as whether to use a GPU device, the
    # random seed, how often to log, and
    # whether we should save the model.
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    # Command to use gpu depending on command line arguments and if there is a cuda device
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Random seed to use
    torch.manual_seed(args.seed)

    # Set to either use gpu or cpu
    device = torch.device("cuda" if use_cuda else "cpu")

    # GPU keywords.
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # Load in the training and testing datasets. Convert to pytorch tensor.
    train_data = DetectionImages(csv_file="../data/labels/box_train_labels.txt", root_dir="../data/train", transform=ToTensor())
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data = DetectionImages(csv_file="../data/labels/box_validation_labels.txt", root_dir="../data/validation", transform=ToTensor())
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # Run model on GPU if available
    model = Net().to(device)

    # Store the lowest test loss found with random search
    lowest_loss = 1000
    # Store the learning curve from lowest test loss
    lowest_test_list = []
    lowest_train_list = []

    # Randomly search over 100 different learning rate and gamma values
    for i in range(100):
        # Boolean value for if this model is the best so far
        best_model = False
        # Get random learning rate
        lr = random.uniform(0.0001, 0.002)
        # Get random gamma
        gamma = random.uniform(0.5, 1)
        print("##################################################")
        print("Learning Rate: ", lr)
        print("Gamma: ", gamma)
        print("##################################################")

        # Specify Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)
        # Store the training and testing losses over time
        train_losses = []
        test_losses = []
        # Run for the set number of epochs. For each epoch, run the training
        # and the testing steps. Scheduler is used to specify the learning rate.
        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
        for epoch in range(1, args.epochs + 1):
            # Train and validate for this epoch
            train_losses = train(args, model, device, train_loader, optimizer, epoch, train_losses)
            test_losses, output = test(args, model, device, test_loader, test_losses)
            scheduler.step()

            # If lowest test loss so far, save model and the training curve
            if lowest_loss > test_losses[epoch - 1]:
                print("New Lowest Loss: ", test_losses[epoch - 1])
                print(output)
                torch.save(model.state_dict(), MODEL_NAME)
                lowest_loss = test_losses[epoch - 1]
                lowest_test_list = test_losses
                lowest_train_list = train_losses
                # Set that this is best model
                best_model = True

        # Save the learning curve if this is best model
        if best_model:
            # Save the learning curve for the best result from random search
            figure, axes = plt.subplots()
            axes.set(xlabel="Epoch", ylabel="Loss", title="Learning Curve")
            axes.plot(np.array(lowest_train_list), label="train_loss", c="b")
            axes.plot(np.array(lowest_test_list), label="validation_loss", c="r")
            plt.legend()
            plt.savefig('box_search_l1_curve.png')
            plt.close()

    # Display the learning curve for the best result from random search
    figure, axes = plt.subplots()
    axes.set(xlabel="Epoch", ylabel="Loss", title="Learning Curve")
    axes.plot(np.array(lowest_train_list), label="train_loss", c="b")
    axes.plot(np.array(lowest_test_list), label="validation_loss", c="r")
    plt.legend()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()