# Name: Daniel Yan
# Email: daniel.yan@vanderbilt.edu
# Description: Predict object location for new image by used network from train.py to predict a label,
# and then converting that label into a floating point value for the center of the label. Takes
# in one command line argument for the path to the image.

# Imports
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

# Constants
MODEL_NAME = "densenet.pt"


def main():
    # Command line arguments for the image path and x and y coordinates
    parser = argparse.ArgumentParser(description='Predict Class for Single Image')
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
    in_transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = in_transform(image)
    # Create tensor of 64 images with all being 0s except first image, since we need
    # test batch size fo 64 for the model
    tensor = torch.tensor((), dtype=torch.float32)
    tensor = tensor.new_zeros((64, 3, 224, 224))
    tensor[0, :, :, :] = image

    # Specify cuda device
    device = torch.device("cuda")
    # Send image to cuda device
    tensor = tensor.to(device, dtype=torch.float32)

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
        # Get the prediction and print
        output = model(tensor)
        print(output.argmax(dim=1, keepdim=True)[0].item())


if __name__ == '__main__':
    main()