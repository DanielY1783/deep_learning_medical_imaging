# Generate predictions for the volumes using the model.

import nibabel as nib
import numpy as np
import os
import torch
from torchvision import models


VAL_IMG_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/data/Testing/img_registered_all/"
SAVE_VOL_PATH = "/content/drive/My Drive/cs8395_deep_learning/assignment3/results/Testing/prediction_float/"
MODEL_NAME = "/content/drive/My Drive/cs8395_deep_learning/assignment3/bin/2d_affine_fixed/deeplabv3_bce_resnet50.pt"

# Start and end indices for where to slice on each axis
X_START = 288
X_END = 512
Y_START = 110
Y_END = 334
Z_START = 75
Z_END = 145

def main():
    # Set to either use gpu or cpu
    device = torch.device("cuda")

    # GPU keywords.
    kwargs = {'num_workers': 1, 'pin_memory': True}

    # Load in saved model
    model = models.segmentation.deeplabv3_resnet50(num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_NAME))

    # Specify that we are in evaluation phase
    model.eval()

    # No gradient calculation because we are in testing phase.
    with torch.no_grad():
        # Iterate through all validation volumes and calculate predicted segmentations for each one.
        for file_name in os.listdir(VAL_IMG_PATH):
            print("Calculating Predictions for", file_name)
            # Load the image
            image = nib.load(VAL_IMG_PATH + file_name)
            # Get the array of values
            image_data = image.get_fdata()
            # Fix zero values added by registration
            image_data = np.where(image_data == 0.0, -1000, image_data)
            # Divide by 1000 to normalize
            image_data = image_data / 1000.0
            # Slice for where the spleen is present
            spleen = image_data[X_START:X_END, Y_START:Y_END, Z_START:Z_END]
            # Put the z axis on the first dimension since each z slice
            # represents a separate image in our 2D model
            spleen = np.transpose(spleen, (2, 0, 1))
            # Convert to torch tensor
            spleen_tensor = torch.from_numpy(spleen)
            # Insert dimension for number of channels
            spleen_tensor = torch.unsqueeze(spleen_tensor, 1)
            # Expand to 3 channels for deeplabv3 architecture
            spleen_tensor = spleen_tensor.expand((Z_END-Z_START, 3, X_END-X_START, Y_END-Y_START))
            spleen_tensor = spleen_tensor.to(device, dtype=torch.float32)

            # Calculate the segmentation output from the model
            segmentation = model(spleen_tensor)["out"]
            # Take out the dimension for number of channels
            segmentation = torch.squeeze(segmentation, 1)
            # Convert to numpy array and transpose again
            segmentation_np = segmentation.cpu().numpy()
            segmentation_np = np.transpose(segmentation_np, (1, 2, 0))

            # Create new numpy array of zeros for the numpy output
            prediction = np.zeros(image_data.shape)
            prediction[X_START:X_END, Y_START:Y_END, Z_START:Z_END] = segmentation_np

            # Save the prediction to file.
            output = nib.Nifti1Image(prediction, image.affine)
            nib.save(output, SAVE_VOL_PATH + file_name)

if __name__ == '__main__':
    main()