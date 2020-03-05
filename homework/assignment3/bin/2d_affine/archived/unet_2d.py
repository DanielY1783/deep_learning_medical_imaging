# Imports for Pytorch
from __future__ import print_function
import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Constants
MODEL_NAME = "unet2d"
# TRAIN_IMG_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train2d/img/"
# TRAIN_LABEL_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train2d/label/"
# VAL_IMG_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val2d/img/"
# VAL_LABEL_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val2d/label/"
TRAIN_IMG_PATH = "../../data/Train2d/img_sample/"
TRAIN_LABEL_PATH = "../../data/Train2d/label_sample/"
VAL_IMG_PATH = "../../data/Val2d/img_sample/"
VAL_LABEL_PATH = "../../data/Val2d/label_sample/"

# Define dataset for image and segmentation mask
class MyDataset(Dataset):
    def __init__(self, image_path, target_path):
        # Create a list of all the names of the files to load
        self.file_names = list(os.listdir(image_path))
        # Create list of images
        self.images_list = []
        self.image_names_list = []
        for file_name in self.file_names:
            # Load in image using numpy
            image = np.load(image_path + file_name)
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image)
            # Insert first dimension for number of channels
            image_tensor = torch.unsqueeze(image_tensor, 0)
            # Add to list of images.
            self.images_list.append(image_tensor)
            self.image_names_list.append(image_path + file_name)
        # Create list of target segmentations
        self.targets_list = []
        self.target_names_list = []
        for file_name in list(os.listdir(image_path)):
            mask = np.load(target_path + file_name)
            # Convert to torch tensor
            mask_tensor = torch.from_numpy(mask)
            # Add to list of masks.
            self.targets_list.append(mask_tensor)
            self.target_names_list.append(image_path + file_name)

    def __getitem__(self, index):
        return self.images_list[index], self.targets_list[index]

    def __len__(self):
        return len(self.images_list)


# Define the UNet Structure
class UNet(nn.Module):
    # Define a single block for the encoder
    def encoder_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=out_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels,
                            out_channels=out_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
        )
        return block

    # Define a single block for the decoder
    def decoder_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels,
                            out_channels=out_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels,
                            out_channels=out_channels, padding=padding),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels)
        )
        return block

    # Define the dimensions for each layer.
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # First encoding block.
        self.encode1 = self.encoder_block(in_channels, 32)
        # First maxpool layer
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        # Second encoding block
        self.encode2 = self.encoder_block(32, 64)
        # Second maxpooling layer
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        # Third encoding block
        self.encode3 = self.encoder_block(64, 128)

        # Middle structure that involves both a maxpool and then a deconv
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv1 = torch.nn.Conv2d(kernel_size=3, in_channels=128,
                                     out_channels=256, padding=1)
        self.conv2 = torch.nn.Conv2d(kernel_size=3, in_channels=256,
                                     out_channels=256, padding=1)
        self.deconv1 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, padding=0, stride=2)

        # First decoder block
        self.decode1 = self.decoder_block(256, 128)
        # Second deconv layer
        self.deconv2 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, padding=0, stride=2)
        # Second decoder block
        self.decode2 = self.decoder_block(128, 64)
        # Third deconv layer
        self.deconv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=2, padding=0, stride=2)
        # Final decoder block
        self.decode3 = self.decoder_block(64, 32)
        # 1x1 convolution to get output
        self.output = torch.nn.Conv2d(kernel_size=1, in_channels=32, out_channels=out_channels)

    # Define forward propagation
    def forward(self, x):
        # First encoder block and maxpool
        # Input: in_channels x 512x512
        # Output: 32x512x512
        encoder1 = self.encode1(x)
        # Input: 32x512x512
        # Output: 32x256x256
        encoder1_mp = self.maxpool1(encoder1)

        # Second encoder block and maxpool
        # Input: 32x256x256
        # Output: 64x256x256
        encoder2 = self.encode2(encoder1_mp)
        # Input: 64x256x256
        # Output: 64x128x128
        encoder2_mp = self.maxpool2(encoder2)

        # Third encoder block
        # Input: 64x128x128
        # Output: 128x128x128
        encoder3 = self.encode3(encoder2_mp)

        # Middle structure with maxpool, convolutions, and then deconv
        # Input: 128x128x128
        # Output: 128x64x64
        encoder3_mp = self.maxpool3(encoder3)
        # Input: 128x64x64
        # Output: 256x64x64
        conv1_output = self.conv1(encoder3_mp)
        # Input: 256x64x64
        # Output: 256x64x64
        conv2_output = self.conv2(conv1_output)
        # Input: 256x64x64
        # Output: 128x128x128
        deconv1_output = self.deconv1(conv2_output)

        # Concatenate encoder 3 output with the deconv1 output and feed to decoder 1
        # Input: 128x128x128 + 128x128x128
        # Output: 256x128x128
        cat1 = torch.cat((encoder3, deconv1_output), dim=1)

        # First decoder block and second deconv layer
        # Input: 256x128x128
        # Output: 128x128x128
        decoder1 = self.decode1(cat1)
        # Input: 128x128x128
        # Output: 64x256x256
        deconv2_output = self.deconv2(decoder1)

        # Concatenate encoder 2 output with the deconv2 output and feed to decoder 2
        # Input: 64x256x256 + 64x256x256
        # Output: 128x256x256
        cat2 = torch.cat((encoder2, deconv2_output), dim=1)

        # Second decoder block and third deconv layer
        # Input: 128x256x256
        # Output: 64x256x256
        decoder2 = self.decode2(cat2)
        # Input: 64x256x256
        # Output: 32x512x512
        deconv3_output = self.deconv3(decoder2)

        # Concatenate encoder 1 output with the deconv3 output and feed to encoder 3
        # Input: 32x512x512 + 32x512x512
        # Output: 64x512x512
        cat3 = torch.cat((encoder1, deconv3_output), dim=1)

        # Feed to last decoder block
        # Input: 64x512x512
        # Output: 32x512x512
        decoder3 = self.decode3(cat3)

        # Final layer to produce 11 segmentation target maps
        # Input: 32x512x512
        # Output: out_channels x 512x512
        y = self.output(decoder3)

        return torch.nn.Softmax(dim=0)(y)


def train(model, device, train_loader, optimizer, epoch, train_losses):
    # Specify that we are in training phase
    model.train()
    # Total Train Loss
    total_loss = 0
    # Iterate through all minibatches.
    for index, (data, target) in enumerate(train_loader):
        # Send training data and the training labels to GPU/CPU
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
        # Zero the gradients carried over from previous step
        optimizer.zero_grad()
        # Obtain the predictions from forward propagation and reshape outputs and labels to calculate
        # cross entropy loss
        output = model(data)

        # Compute the cross entropy for the loss and update total loss.
        loss = F.cross_entropy(output, target)
        total_loss += loss.item()
        # Perform backward propagation to compute the negative gradient, and
        # update the gradients with optimizer.step()
        loss.backward()
        optimizer.step()

    # Update training error and add to accumulation of training loss over time.
    train_error = total_loss / len(train_loader)
    train_losses.append(train_error)
    # Print out the epoch and train loss
    print("############################################################################")
    print("Train Epoch: ", epoch)
    print("############################################################################")
    print("Average Training Loss: ", train_error)
    return train_losses


def test(model, device, test_loader, test_losses):
    # Specify that we are in evaluation phase
    model.eval()
    # Set the loss and number of correct instances initially to 0.
    test_loss = 0
    # Sum of true positives, false positives, true negatives, and false negatives
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # For each testing example, we run forward
        # propagation to calculate the
        # testing prediction. Update the total loss
        # and f1 score with counters from above
        for index, (data, target) in enumerate(test_loader):
            # Send training data and the training labels to GPU/CPU
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
            # Obtain the output from the model
            output = model(data)
            # Calculate the loss using cross entropy.
            loss = F.cross_entropy(output, target)
            # Increment the total test loss
            test_loss += loss.item()

            # Get the prediction by getting the index with the maximum probability
            pred = output.argmax(dim=1, keepdim=True).cpu().numpy()
            # Filter both the prediction and the target by only class 1 for spleen
            pred_filtered = np.where(pred == 1, 1, 0)
            target_filtered = np.where(target.cpu().numpy() == 1, 1, 0)

            # Calculate the true positives, false positives, true negatives, and false negatives
            # and increment total sums
            true_positives = float(np.sum(np.where(np.logical_and(pred_filtered == 1, target_filtered == 1), 1, 0)))
            false_positives = float(np.sum(np.where(np.logical_and(pred_filtered == 1, target_filtered == 0), 1, 0)))
            true_negatives = float(np.sum(np.where(np.logical_and(pred_filtered == 0, target_filtered == 0), 1, 0)))
            false_negatives = float(np.sum(np.where(np.logical_and(pred_filtered == 0, target_filtered == 1), 1, 0)))
            total_tp += true_positives
            total_tn += true_negatives
            total_fp += false_positives
            total_fn += false_negatives

        # Calculate precision, recall, and f1 and print out statistics for validation set
        print("Average Validation Loss: ", test_loss / len(test_loader))
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1 = precision * recall * 2 / (precision + recall)
        print("Average Validation Precision: ", precision)
        print("Average Validation Recall: ", recall)
        print("Average Validation F1: ", f1)
        print("Total Validation True Positives: ", total_tp)
        print("Total Validation True Negatives: ", total_tn)
        print("Total Validation False Positives: ", total_fp)
        print("Total Validation False Negatives: ", total_fn)

    # Append test loss to total losses
    test_losses.append(test_loss / len(test_loader))
    return test_losses

# Main structure
def main():
    print("Entering Main")
    # Command line arguments for hyperparameters of model/training.
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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

    # Load in the dataset
    train_data = MyDataset(image_path=TRAIN_IMG_PATH, target_path=TRAIN_LABEL_PATH)
    val_data = MyDataset(image_path=VAL_IMG_PATH, target_path=VAL_LABEL_PATH)
    # Create data loader for training and validation
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    print("Finished Loading Data")

    # Send model to gpu
    model = UNet(1, 14).to(device)
    # Specify Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Store training and validation losses over time
    train_losses = []
    val_losses = []

    # Create scheduler.
    scheduler = StepLR(optimizer, step_size=1)

    # Store the lowest loss found so far for early stopping
    lowest_loss = 1000

    # Train the model for the set number of epochs
    for epoch in range(1, args.epochs + 1):
        # Train and validate for this epoch
        train_losses = train(model, device, train_loader, optimizer, epoch, train_losses)
        val_losses = test(model, device, val_loader, val_losses)
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
