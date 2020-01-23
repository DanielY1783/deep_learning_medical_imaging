# Imports
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# Constants for number of pixels in each image.
X_PIXELS = 490
Y_PIXELS = 326

def main():
    # Command line arguments for the image path and x and y coordinates
    parser = argparse.ArgumentParser(description='Visualize a Single Prediction Location')
    parser.add_argument('image_path', help='path to the image to display')
    parser.add_argument("x_cord", type=float, help="x coordinate for the object")
    parser.add_argument("y_cord", type=float, help="y coordinate for the object")
    args = parser.parse_args()
    # Open the image passed by the command line argument
    image_path = Image.open(args.image_path)
    x_cord = args.x_cord
    y_cord = args.y_cord
    # Show the image and labels
    plt.imshow(image_path)
    plt.scatter(x_cord*X_PIXELS, y_cord*Y_PIXELS, s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

if __name__ == '__main__':
    main()