# Imports for Pytorch
from __future__ import print_function
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
from skimage import io

# Constants for the name of the model to save to
MODEL_NAME = "densenet_no_class_1"

# Class for the dataset
class ImagesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = pd.read_csv(csv_file, sep="\t", header=None)
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
        # Normalize images with mean and standard deviation for pretrained models
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        in_transform = transforms.Compose([normalize])
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image = in_transform(image)
        # Format label as torch tensor
        label = torch.from_numpy(np.array(label).astype(int))
        return {'image': image,
                'label': label}


def train(args, model, device, train_loader, optimizer, epoch, train_losses):
    # Specify that we are in training phase
    model.train()
    # Total Train Loss
    total_loss = 0
    # Iterate through all minibatches.
    for batch_index, batch_sample in enumerate(train_loader):
        # Send training data and the training labels to GPU/CPU
        data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device, dtype=torch.long)
        # Zero the gradients carried over from previous step
        optimizer.zero_grad()
        # Get the label
        target = target[:, 0]
        # Obtain the predictions from forward propagation
        output = model(data)
        # Compute the cross entropy for the loss.
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        # Perform backward propagation to compute the negative gradient, and
        # update the gradients with optimizer.step()
        loss.backward()
        optimizer.step()
    # Update training error and add to accumulation of training loss over time.
    train_error = total_loss / len(train_loader)
    train_losses.append(train_error)
    # Print output if epoch is finished
    print('Train Epoch: {} \tAverage Loss: {:.6f}'.format(epoch, train_error))



def test(args, model, device, test_loader, test_losses):
    # Specify that we are in evaluation phase
    model.eval()
    # Set the loss and number of correct instances initially to 0.
    test_loss = 0
    # Set no correct predictions initially
    correct = 0
    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # For each testing example, we run forward
        # propagation to calculate the
        # testing prediction. Update the total loss
        # and the number of correct predictions
        # with the counters from above.
        for batch_idx, batch_sample in enumerate(test_loader):
            # Send data and the labels to GPU/CPU
            data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device,
                                                                                          dtype=torch.long)
            # Get the label with one less dimension
            target = target[:, 0]
            # Obtain the output from the model
            output = model(data)
            # Calculate the loss using cross entropy.
            loss = F.cross_entropy(output, target)
            # Increment the total test loss
            test_loss += loss.item()
            # Get the prediction by getting the index with the maximum probability
            pred = output.argmax(dim=1, keepdim=True)
            # Get the number of correct predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Append test loss to total losses
    test_losses.append(test_loss / len(test_loader))

    # Print out the statistics for the testing set.
    print('\nTest set: Average loss: {:.6f}'.format(
        test_loss / len(test_loader)))
    # Print out the number of correct predictions
    print('\nTest set: Correct Predictions: {}/{}'.format(
        correct, len(test_loader.dataset)))
    # Print out testing accuracy
    print("\nTest set: Accuracy: {}".format(float(correct/len(test_loader.dataset))))



def main():
    # Command line arguments for hyperparameters of model/training.
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=25, metavar='N',
                        help='number of epochs to train (default: 25)')
    parser.add_argument('--gamma', type=float, default=1, metavar='N',
                        help='gamma value for learning rate decay (default: 1)')
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

    # Load in the dataset and split into training and validation
    data = ImagesDataset(csv_file="../data/labels/no_class_1_train_labels.csv", root_dir="../data/resized224/train/", transform=ToTensor())
    train_size = int(0.9 * len(data))
    test_size = len(data) - train_size
    train_data, val_data = torch.utils.data.random_split(data, [train_size, test_size])
    # Create data loader for training and validation
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

    # Use densenet
    model = models.densenet121()
    # Number of classes is 6 since we don't have class 1 anymore
    num_classes = 6
    # Reshape the output for densenet for this problem
    model.classifier = nn.Linear(1024, num_classes)
    # Send model to gpu
    model = model.to(device)
    # Specify Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Store training and validation losses over time
    train_losses = []
    val_losses = []

    # Create scheduler.
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Store the lowest loss found so far for early stopping
    lowest_loss = 1000

    # Train the model for the set number of epochs
    for epoch in range(1, args.epochs + 1):
        # Train and validate for this epoch
        train(args, model, device, train_loader, optimizer, epoch, train_losses)
        test(args, model, device, val_loader, val_losses)
        scheduler.step()

        # If we find the lowest loss so far, store the model and learning curve
        if lowest_loss > val_losses[epoch - 1]:
            # Update the lowest loss
            lowest_loss = val_losses[epoch - 1]
            print("New lowest validation loss: ", lowest_loss)

            # Create learning curve
            figure, axes = plt.subplots()
            # Set axes labels and title
            axes.set(xlabel="Epoch", ylabel="Loss", title="Learning Curve")
            # Plot the learning curves for training and validation loss
            axes.plot(np.array(train_losses), label="train_loss", c="b")
            axes.plot(np.array(val_losses), label="validation_loss", c="r")
            plt.legend()
            # Save the figure
            plt.savefig(MODEL_NAME + ".png")
            plt.close()

            # Save the model
            torch.save(model.state_dict(), MODEL_NAME + ".pt")

if __name__ == '__main__':
    main()