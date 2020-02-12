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
MODEL_NAME_2 = "densenet_no_class_1_pretrained.pt"

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

    # Create first densenet model that predicts if image is class 1 or not
    model_binary = models.densenet121()
    # Reshape the output for densenet for this problem
    model_binary.classifier = nn.Linear(1024, 7)
    # Send model to gpu and load in saved parameters for prediction
    model_binary = model_binary.to(device)
    model_binary.load_state_dict(torch.load(MODEL_NAME_1))
    # Specify that we are in evaluation phase
    model_binary.eval()

    # Create second densenet model that predicts class for all classes except class 1
    model_no_class_1 = models.densenet121()
    # Reshape the output for densenet for this problem
    model_no_class_1.classifier = nn.Linear(1024, 6)
    # Send model to gpu and load in saved parameters for prediction
    model_no_class_1 = model_no_class_1.to(device)
    model_no_class_1.load_state_dict(torch.load(MODEL_NAME_2))
    # Specify that we are in evaluation phase
    model_no_class_1.eval()

    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # Accumulate the predictions for each model and actual labels
        predictions_binary = torch.tensor((), dtype=torch.long).to(device)
        predictions_no_class_1 = torch.tensor((), dtype=torch.long).to(device)
        actual = torch.tensor((), dtype=torch.long).to(device)

        # Iterate through all batches.
        for batch_idx, batch_sample in enumerate(test_loader):
            # Send data and the labels to GPU/CPU
            data, target = batch_sample["image"].to(device, dtype=torch.float32), batch_sample["label"].to(device,
                                                                                                           dtype=torch.long)
            # Get the label with one less dimension
            target = target[:, 0]
            # Predict the current batch for both models
            output_binary = model_binary(data)
            output_no_class_1 = model_no_class_1(data)
            # Get the maximum probability from softmax, and slice to get rid of unneeded dimension.
            output_binary = output_binary.argmax(dim=1, keepdim=True)[:, 0]
            output_no_class_1 = output_no_class_1.argmax(dim=1, keepdim=True)[:, 0]
            # Append prediction and actual values to cumulative predictions.
            predictions_binary = torch.cat((predictions_binary, output_binary), 0)
            predictions_no_class_1 = torch.cat((predictions_no_class_1, output_no_class_1), 0)
            actual = torch.cat((actual, target), 0)

        # Convert to numpy array
        predictions_binary = predictions_binary.cpu().numpy()
        predictions_no_class_1 = predictions_no_class_1.cpu().numpy()
        actual = actual.cpu().numpy()

        # Change classes for the predictions without class 1 back to original classes by adding
        # 1 to all values except for class 0
        predictions_no_class_1 = np.where(predictions_no_class_1 > 0, predictions_no_class_1 + 1, 0)

        # Change the predictions from first model for binary of if prediction is 1 or 0
        predictions_binary = np.where(predictions_binary == 1, 1, 0)

        # Generate prediction by using class 1 if the binary model predicts class 1, and the class
        # predicted by the model without class 1 if the binary model predicts that the image
        # is not class 1.
        predictions = np.where(predictions_binary == 1, 1, predictions_no_class_1)

        # Use scikit-learn to print out accuracy, precision, and recall
        print("Test set accuracy: ", metrics.accuracy_score(actual, predictions))
        print("Test set precision: ", metrics.precision_score(actual, predictions, average="weighted"))
        print("Test set recall: ", metrics.recall_score(actual, predictions, average="weighted"))


        # Use scikit-learn to calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(actual, predictions, normalize="true")
        # Use seaborn to plot heatmap
        axes = seaborn.heatmap(confusion_matrix, annot=True)
        axes.set(xlabel="Predicted Label", ylabel="Actual Label", title="Confusion Matrix")
        # Save as image and show plot.
        plt.savefig("confusion_matrix_double_densenet.png")
        plt.show()


        labels_binary = np.where(actual == 1, 1, 0)
        print(metrics.accuracy_score(labels_binary, predictions_binary))
        # Use scikit-learn to calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(labels_binary, predictions_binary, normalize="true")
        # Use seaborn to plot heatmap
        axes = seaborn.heatmap(confusion_matrix, annot=True)
        axes.set(xlabel="Predicted Label", ylabel="Actual Label", title="Confusion Matrix")
        # Save as image and show plot.
        plt.savefig("confusion_matrix_binary.png")
        plt.show()

if __name__ == '__main__':
    main()