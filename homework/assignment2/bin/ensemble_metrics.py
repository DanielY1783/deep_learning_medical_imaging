# Name: Daniel Yan
# Email: daniel.yan@vanderbilt.edu
# Description: Predict object location for new image by used network from train.py to predict a label,
# and then converting that label into a floating point value for the center of the label. Takes
# in one command line argument for the path to the image.

# Imports
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn import metrics
from skimage import io
import seaborn

# Constants
MODEL_NAME_1 = "densenet_pretrained.pt"
MODEL_NAME_2 = "resnet_pretrained.pt"

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


def main():
    # Load in the test dataset
    data = ImagesDataset(csv_file="../data/labels/formatted_test_labels.csv", root_dir="../data/resized224/test/",
                         transform=ToTensor())
    # Create data loader for batch testing
    test_loader = DataLoader(data, batch_size=64, shuffle=False, num_workers=0)

    # Specify cuda device
    device = torch.device("cuda")

    # Create first densenet model
    model_densenet = models.densenet121()
    model_densenet.classifier = nn.Linear(1024, 7)
    # Send model to gpu and load in saved parameters for prediction
    model_densenet = model_densenet.to(device)
    model_densenet.load_state_dict(torch.load(MODEL_NAME_1))
    # Specify that we are in evaluation phase
    model_densenet.eval()

    # Create second resnet model
    model_resnet = models.resnet50()
    model_resnet.fc = nn.Linear(512, 7)
    # Send model to gpu and load in saved parameters for prediction
    model_resnet = model_resnet.to(device)
    model_resnet.load_state_dict(torch.load(MODEL_NAME_2))
    # Specify that we are in evaluation phase
    model_resnet.eval()

    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # Accumulate the predictions for each model and actual labels
        predictions_densenet = torch.tensor((), dtype=torch.long).to(device)
        predictions_resnet = torch.tensor((), dtype=torch.long).to(device)
        actual = torch.tensor((), dtype=torch.long).to(device)

        # Iterate through all batches.
        for batch_idx, batch_sample in enumerate(test_loader):
            # Send data and the labels to GPU/CPU
            data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device,
                                                                                                           dtype=torch.long)
            # Get the label with one less dimension
            target = target[:, 0]
            # Predict the current batch for both models
            output_densenet = model_densenet(data)
            output_resnet = model_resnet(data)

            # Append prediction and actual values to cumulative predictions.
            predictions_densenet = torch.cat((predictions_densenet, output_densenet), 0)
            predictions_resnet = torch.cat((predictions_resnet, output_resnet), 0)
            actual = torch.cat((actual, target), 0)

        # Convert to numpy array
        predictions_densenet = predictions_densenet.cpu().numpy()
        predictions_resnet = predictions_resnet.cpu().numpy()
        actual = actual.cpu().numpy()

        # Add up probabilities from both predictions
        predictions = predictions_densenet + predictions_resnet

        # Get up maxiumum class prediction
        predictions = np.argmax(predictions, axis=0)

        print(predictions)
        # # Use scikit-learn to print out accuracy, precision, and recall
        # print("Test set accuracy: ", metrics.accuracy_score(actual, predictions))
        # print("Test set precision: ", metrics.precision_score(actual, predictions, average="weighted"))
        # print("Test set recall: ", metrics.recall_score(actual, predictions, average="weighted"))
        #
        #
        # # Use scikit-learn to calculate confusion matrix
        # confusion_matrix = metrics.confusion_matrix(actual, predictions, normalize="true")
        # # Use seaborn to plot heatmap
        # axes = seaborn.heatmap(confusion_matrix, annot=True)
        # axes.set(xlabel="Predicted Label", ylabel="Actual Label", title="Confusion Matrix")
        # # Save as image and show plot.
        # plt.savefig("confusion_matrix_new.png")
        # plt.show()
        #
        #
        # labels_binary = np.where(actual == 1, 1, 0)
        # print(metrics.accuracy_score(labels_binary, predictions_densenet))
        # # Use scikit-learn to calculate confusion matrix
        # confusion_matrix = metrics.confusion_matrix(labels_binary, predictions_densenet, normalize="true")
        # # Use seaborn to plot heatmap
        # axes = seaborn.heatmap(confusion_matrix, annot=True)
        # axes.set(xlabel="Predicted Label", ylabel="Actual Label", title="Confusion Matrix")
        # # Save as image and show plot.
        # plt.savefig("confusion_matrix_binary.png")
        # plt.show()

if __name__ == '__main__':
    main()