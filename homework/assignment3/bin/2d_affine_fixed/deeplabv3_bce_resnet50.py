# Author: Daniel Yan
# Email: daniel.yan@vanderbilt.edu
# Description: Train deeplabv3 for segmentation

import argparse
from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import StepLR

# Constants
MODEL_NAME = "/content/drive/My Drive/cs8395_deep_learning/assignment3/bin/2d_affine_fixed/deeplabv3_bce_resnet50"
TRAIN_IMG_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/affine_fixed/img_cropped/"
TRAIN_LABEL_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Train/affine_fixed/label_cropped_filtered/"
VAL_IMG_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/affine_fixed/img_cropped/"
VAL_LABEL_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Val/affine_fixed/label_cropped_filtered/"

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
            image_tensor_expanded = image_tensor.expand((3, 224, 224))
            # Add to list of images.
            self.images_list.append(image_tensor_expanded)
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

def train(model, device, train_loader, optimizer, epoch, train_losses):
    # Specify that we are in training phase
    model.train()
    # Total Train Loss
    total_loss = 0
    # Iterate through all minibatches.
    for index, (data, target) in enumerate(train_loader):
        # Send training data and the training labels to GPU/CPU
        data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
        # Zero the gradients carried over from previous step
        optimizer.zero_grad()
        # Obtain the predictions from forward propagation
        output = model(data)["out"]
        output = torch.squeeze(output, 1)
        # Compute the cross entropy for the loss and update total loss.
        loss = torch.nn.BCEWithLogitsLoss()(output, target)
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
    # Create dictionary of predictions for different thresholds.
    # Initialize sums of true positives, true negatives, false positives, and false negatives to 0
    threshold_dict = {}
    for threshold in [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]:
        threshold_dict[str(threshold)] = {}
        threshold_dict[str(threshold)]["total_tp"] = 0
        threshold_dict[str(threshold)]["total_tn"] = 0
        threshold_dict[str(threshold)]["total_fp"] = 0
        threshold_dict[str(threshold)]["total_fn"] = 0
    # Specify that we are in evaluation phase
    model.eval()
    # Set the loss and number of correct instances initially to 0.
    test_loss = 0
    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # For each testing example, we run forward
        # propagation to calculate the
        # testing prediction. Update the total loss
        # and f1 score with counters from above
        for index, (data, target) in enumerate(test_loader):
            # Send training data and the training labels to GPU/CPU
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
            # Obtain the output from the model
            output = model(data)["out"]
            output = torch.squeeze(output, 1)
            # Calculate the loss using cross entropy.
            loss = torch.nn.BCEWithLogitsLoss()(output, target)
            # Increment the total test loss
            test_loss += loss.item()

            # Convert output to numpy array
            output = output.cpu().numpy()
            # Calculate stats for each threshold
            for threshold in [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]:
                # Filter both the prediction and the target by only class 1 for spleen
                pred_filtered = np.where(output > threshold, 1, 0)
                target_filtered = np.where(target.cpu().numpy() == 1, 1, 0)

                # Calculate the true positives, false positives, true negatives, and false negatives
                # and increment total sums
                true_positives = float(np.sum(np.where(np.logical_and(pred_filtered == 1, target_filtered == 1), 1, 0)))
                false_positives = float(np.sum(np.where(np.logical_and(pred_filtered == 1, target_filtered == 0), 1, 0)))
                true_negatives = float(np.sum(np.where(np.logical_and(pred_filtered == 0, target_filtered == 0), 1, 0)))
                false_negatives = float(np.sum(np.where(np.logical_and(pred_filtered == 0, target_filtered == 1), 1, 0)))
                threshold_dict[str(threshold)]["total_tp"] += true_positives
                threshold_dict[str(threshold)]["total_tn"] += true_negatives
                threshold_dict[str(threshold)]["total_fp"] += false_positives
                threshold_dict[str(threshold)]["total_fn"] += false_negatives

        # Calculate precision, recall, and f1 and print out statistics for validation set
        print("Average Validation Loss: ", test_loss / len(test_loader))
        # Find results for each threshold.
        for threshold in [-0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5]:
            print("At threshold ", str(threshold))
            print("Total Validation True Positives: ", threshold_dict[str(threshold)]["total_tp"])
            print("Total Validation True Negatives: ", threshold_dict[str(threshold)]["total_tn"])
            print("Total Validation False Positives: ", threshold_dict[str(threshold)]["total_fp"])
            print("Total Validation False Negatives: ", threshold_dict[str(threshold)]["total_fn"])

            if (threshold_dict[str(threshold)]["total_tp"] > 0 and threshold_dict[str(threshold)]["total_fp"]> 0
                    and threshold_dict[str(threshold)]["total_fn"] > 0):
                precision = threshold_dict[str(threshold)]["total_tp"] / (threshold_dict[str(threshold)]["total_tp"]
                                                                     + threshold_dict[str(threshold)]["total_fp"])
                recall = threshold_dict[str(threshold)]["total_tp"] / (threshold_dict[str(threshold)]["total_tp"] +
                                                                  threshold_dict[str(threshold)]["total_fn"])
                f1 = 2 * precision * recall / (precision + recall)
                print("Precision: ", precision)
                print("Recall: ", recall)
                print("F1: ", f1)

    # Append test loss to total losses
    test_losses.append(test_loss / len(test_loader))
    return test_losses

# Main structure
def main():
    print("Entering Main")
    # Command line arguments for hyperparameters of model/training.
    parser = argparse.ArgumentParser(description='PyTorch Object Detection')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()
    # Command to use gpu
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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    print("Finished Loading Data")

    # Send model to gpu
    model = models.segmentation.deeplabv3_resnet50(num_classes=1).to(device)
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

        # If we find the lowest loss so far, store the model and learning curve
        if lowest_loss > val_losses[epoch - 1]:
            # Update the lowest loss
            lowest_loss = val_losses[epoch - 1]
            print("New lowest validation loss: ", lowest_loss)

            # Save the model
            torch.save(model.state_dict(), MODEL_NAME + ".pt")


if __name__ == '__main__':
    main()
