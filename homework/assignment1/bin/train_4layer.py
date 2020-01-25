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
MODEL_NAME_X = "network_softmax_4layer_x.pt"
MODEL_NAME_Y = "network_softmax_4layer_y.pt"

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
        # Convolutional Layers with batch normalization
        self.conv1 = nn.Conv2d(3, 10, 3, 1)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, 1)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(40)
        self.conv3 = nn.Conv2d(40, 80, 3, 1)
        self.conv3_bn = nn.BatchNorm2d(80)

        # Dropout values for convolutional and fully connected layers
        self.dropout1 = nn.Dropout2d(0.4)
        self.dropout2 = nn.Dropout2d(0.4)

        # Two fully connected layers. Input is 2347380 because 243x161x60
        # as shown in the forward part.
        self.fc1 = nn.Linear(171680, 128)
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
        # Output dimensions: 244x162x10
        x = F.max_pool2d(x, 2)

        # Input dimensions: 244x162x10
        # Output dimensions: 242x160x20
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 242x160x20
        # Output dimensions: 121x80x20
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 121x80x20
        # Output dimensions: 119x78x20
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 119x78x20
        # Output dimensions: 60x39x40
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 60x39x40
        # Output dimensions: 58x37x80
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = F.relu(x)
        x = self.dropout1(x)
        # Input dimensions: 58x37x80
        # Output dimensions: 29x19x80
        x = F.max_pool2d(x, 2, ceil_mode=True)

        # Input dimensions: 29x19x80
        # Output dimensions: 171680x1
        x = torch.flatten(x, 1)
        # Input dimensions: 171680x1
        # Output dimensions: 128x1
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout2(x)
        # Input dimensions: 128x1
        # Output dimensions: 20x1
        x = self.fc2(x)
        output = torch.softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch, train_losses):
    # Specify that we are in training phase
    model.train()
    # Total Train Loss
    total_loss = 0
    # Iterate through all minibatches.
    for batch_idx, batch_sample in enumerate(train_loader):
        # Send training data and the training labels to GPU/CPU
        data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device, dtype=torch.long)
        # Zero the gradients carried over from previous step
        optimizer.zero_grad()
        target = target.squeeze_()
        # Obtain the predictions from forward propagation
        output = model(data)
        # Compute the cross entropy for loss
        loss = F.cross_entropy(output, target)
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
    correct = 0
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
                                                                                          dtype=torch.long)
            target = target.squeeze_()
            output = model(data)
            loss = F.cross_entropy(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Average the loss by dividing by the total number of testing instances and add to accumulation of losses.
    test_error = test_loss / len(test_loader.dataset)
    test_losses.append(test_error)

    # Print out the statistics for the testing set.
    print('\nTest set: Average loss: {:.6f}\n'.format(
        test_error))
    print("Correct instances: ", correct)

    # Return accumulated test losses over epochs
    return test_losses, pred


def main():
    # Command line arguments for hyperparameters of
    # training and testing batch size, the number of
    # epochs, the learning rate, gamma, and other
    # settings such as whether to use a GPU device, the
    # random seed, how often to log, and
    # whether we should save the model.
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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

    # Load in the training and testing datasets for the x values. Convert to pytorch tensor.
    train_data_x = DetectionImages(csv_file="../data/labels/x_class_train_labels.txt", root_dir="../data/train", transform=ToTensor())
    train_loader_x = DataLoader(train_data_x, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data_x = DetectionImages(csv_file="../data/labels/x_class_validation_labels.txt", root_dir="../data/validation", transform=ToTensor())
    test_loader_x = DataLoader(test_data_x, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # Load in the training and testing datasets for the y values. Convert to pytorch tensor.
    train_data_y = DetectionImages(csv_file="../data/labels/y_class_train_labels.txt", root_dir="../data/train", transform=ToTensor())
    train_loader_y = DataLoader(train_data_y, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_data_y = DetectionImages(csv_file="../data/labels/y_class_validation_labels.txt", root_dir="../data/validation", transform=ToTensor())
    test_loader_y = DataLoader(test_data_y, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # Create model for x prediction
    model_x = Net().to(device)
    # Create model for y prediction
    model_y = Net().to(device)

    # Store the lowest test loss found with random search for both x and y models
    lowest_loss_x = 1000
    lowest_loss_y = 1000
    # Store the learning curve from lowest test loss for x and y models
    lowest_test_list_x = []
    lowest_train_list_x = []
    lowest_test_list_y = []
    lowest_train_list_y = []

    # Randomly search over 30 different learning rate and gamma value combinations
    for i in range(30):
        # Boolean variable for if this model for either x or y is the best so far
        best_model_x = False
        best_model_y = False
        # Get random learning rate
        lr = random.uniform(0.0005, 0.002)
        # Get random gamma
        gamma = random.uniform(0.5, 1)
        # Print out the current learning rate and gamma value
        print("##################################################")
        print("Learning Rate: ", lr)
        print("Gamma: ", gamma)
        print("##################################################")

        # Specify Adam optimizer for x and y models
        optimizer_x = optim.Adam(model_x.parameters(), lr=lr)
        optimizer_y = optim.Adam(model_y.parameters(), lr=lr)

        # Store the training and testing losses over time
        train_losses_x = []
        test_losses_x = []
        train_losses_y = []
        test_losses_y = []
        # Create schedulers for x and y models.
        scheduler_x = StepLR(optimizer_x, step_size=1, gamma=gamma)
        scheduler_y = StepLR(optimizer_x, step_size=1, gamma=gamma)


        # Train the x model for the set number of epochs
        print("===========Training X Model================")
        for epoch in range(1, args.epochs + 1):
            # Train and validate for this epoch
            train_losses_x = train(args, model_x, device, train_loader_x, optimizer_x, epoch, train_losses_x)
            test_losses_x, output_x = test(args, model_x, device, test_loader_x, test_losses_x)
            scheduler_x.step()

            # If this is the lowest validation loss so far, save model and the training curve. This allows
            # us to recover a model for early stopping
            if lowest_loss_x > test_losses_x[epoch - 1]:
                print("New Lowest Loss For X Model: ", test_losses_x[epoch - 1])
                print("Validation Predictions: ")
                print(output_x)
                torch.save(model_x.state_dict(), MODEL_NAME_X)
                lowest_loss_x = test_losses_x[epoch - 1]
                lowest_test_list_x = test_losses_x
                lowest_train_list_x = train_losses_x
                # Set that this is best model
                best_model_x = True

        # Train the y model for the set number of epochs
        print("===========Training Y Model================")
        for epoch in range(1, args.epochs + 1):
            # Train and validate for this epoch
            train_losses_y = train(args, model_y, device, train_loader_y, optimizer_y, epoch, train_losses_y)
            test_losses_y, output_y = test(args, model_y, device, test_loader_y, test_losses_y)
            scheduler_y.step()

            # If this is the lowest validation loss so far, save model and the training curve. This allows
            # us to recover a model for early stopping
            if lowest_loss_y > test_losses_y[epoch - 1]:
                print("New Lowest Loss For Y Model: ", test_losses_y[epoch - 1])
                print("Validation Predictions: ")
                print(output_y)
                torch.save(model_y.state_dict(), MODEL_NAME_Y)
                lowest_loss_y = test_losses_y[epoch - 1]
                lowest_test_list_y = test_losses_y
                lowest_train_list_y = train_losses_y
                # Set that this is best model
                best_model_y = True

        # Save the learning curve if this is best model
        if best_model_x:
            # Save the learning curve for the best result from random search
            figure, axes = plt.subplots()
            axes.set(xlabel="Epoch", ylabel="Loss For X Model", title="Learning Curve For X Model")
            axes.plot(np.array(lowest_train_list_x), label="train_loss", c="b")
            axes.plot(np.array(lowest_test_list_x), label="validation_loss", c="r")
            plt.legend()
            plt.savefig('curve_4layer_x.png')
            plt.close()

        if best_model_y:
            # Save the learning curve for the best result from random search
            figure, axes = plt.subplots()
            axes.set(xlabel="Epoch", ylabel="Loss For Y Model", title="Learning Curve For Y Model")
            axes.plot(np.array(lowest_train_list_y), label="train_loss", c="b")
            axes.plot(np.array(lowest_test_list_y), label="validation_loss", c="r")
            plt.legend()
            plt.savefig('curve_4layer_y.png')
            plt.close()



    # Display the learning curves for the best results from random search
    figure, axes = plt.subplots()
    axes.set(xlabel="Epoch", ylabel="Loss For X Model", title="Learning Curve For X Model")
    axes.plot(np.array(lowest_train_list_x), label="train_loss", c="b")
    axes.plot(np.array(lowest_test_list_x), label="validation_loss", c="r")
    plt.legend()
    plt.show()
    plt.close()

    figure, axes = plt.subplots()
    axes.set(xlabel="Epoch", ylabel="Loss For Y Model", title="Learning Curve For Y Model")
    axes.plot(np.array(lowest_train_list_y), label="train_loss", c="b")
    axes.plot(np.array(lowest_test_list_y), label="validation_loss", c="r")
    plt.legend()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()