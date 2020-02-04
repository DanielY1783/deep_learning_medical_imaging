# Name: Daniel Yan
# Email: daniel.yan@vanderbilt.edu
# Description: Predict object location for new image by used network from train.py to predict a label,
# and then converting that label into a floating point value for the center of the label. Takes
# in one command line argument for the path to the image.

# Imports
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn import metrics
from skimage import io

# Constants
MODEL_NAME = "densenet.pt"

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

    # Use densenet
    model = models.densenet121()
    # Number of classes is 7
    num_classes = 7
    # Reshape the output for densenet for this problem
    model.classifier = nn.Linear(1024, num_classes)
    # Send model to gpu and load in saved parameters for prediction
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_NAME))
    # Specify that we are in evaluation phase
    model.eval()

    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # Accumulate the predictions and actual labels
        predictions = torch.tensor((), dtype=torch.long).to(device)
        actual = torch.tensor((), dtype=torch.long).to(device)

        # Iterate through all batches.
        for batch_idx, batch_sample in enumerate(test_loader):
            # Send data and the labels to GPU/CPU
            data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device,
                                                                                                           dtype=torch.long)
            # Get the label with one less dimension
            target = target[:, 0]
            # Predict the current batch
            output = model(data)
            # Get the maximum probability from softmax, and slice to get rid of unneeded dimension.
            output = output.argmax(dim=1, keepdim=True)[:, 0]
            # Append prediction and actual values to cumulative predictions.
            predictions = torch.cat((predictions, output), 0)
            actual = torch.cat((actual, target), 0)

        # Convert to numpy array
        predictions = predictions.cpu().numpy()
        actual = actual.cpu().numpy()

        # Use scikit-learn to print out accuracy score
        print(len(actual))
        print("Test set accuracy: ", metrics.accuracy_score(actual, predictions))
        print("Test set precision: ", metrics.precision_score(actual, predictions, average="weighted"))
        print("Test set recall: ", metrics.recall_score(actual, predictions, average="weighted"))

if __name__ == '__main__':
    main()