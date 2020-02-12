# Author: Daniel Yan
# Email: daniel.yan@vanderbilt.edu
# Description: Quick script to resize images to 224x224 to use torchvision models.

from PIL import Image
import os, sys

# Constants
OLD_PATH = "../data/original/"
NEW_PATH = "../data/resized224/"

# Resize training images
for file_name in os.listdir(OLD_PATH+"train"):
    # Open image
    old_image = Image.open(OLD_PATH+"train/"+file_name)
    # Resize image
    new_image = old_image.resize((224, 224), Image.ANTIALIAS)
    # Save image
    new_image.save(NEW_PATH+"train/"+file_name)

# Resize testing images
for file_name in os.listdir(OLD_PATH+"test"):
    # Open image
    old_image = Image.open(OLD_PATH+"test/"+file_name)
    # Resize image
    new_image = old_image.resize((224, 224), Image.ANTIALIAS)
    # Save image
    new_image.save(NEW_PATH+"test/"+file_name)