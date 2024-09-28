import numpy as np
import pathlib

def get_dataSet():
    
    # Load the MNIST data from the .npz file
    with np.load("mnist.npz") as f:
        images = f["x_train"]
        labels = f["y_train"]

    # Normalize image data to range [0, 1] and flatten each image to a 1D array
    images = images.astype(np.float32) / 255.0
    images = images.reshape(images.shape[0], -1)  # Flatten each image (28x28 -> 784)

    # Convert labels to one-hot encoded format
    labels = np.eye(10)[labels]

    return images, labels
