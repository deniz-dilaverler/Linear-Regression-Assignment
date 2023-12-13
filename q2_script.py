import matplotlib.pyplot as plt
import numpy as np
import gzip


def __get_encoded_labels(label_data):
    number_labels = len(np.unique(label_data))
    encoded_labels = np.zeros((len(label_data), number_labels), dtype=int)
    return encoded_labels


# One-hot encoding of the labels
def one_hot_encoding(label_data):
    encoded_labels = __get_encoded_labels(label_data)

    for i, label in enumerate(label_data):
        encoded_labels[i, label] = 1
    return encoded_labels


# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), "B", offset=16).astype("float32")
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28 * 28)
    return flattened_pixels


# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), "B", offset=8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels


# Function to read the entire dataset
def read_dataset():
    X_train = read_pixels("./data/train-images-idx3-ubyte.gz")
    y_train = read_labels("./data/train-labels-idx1-ubyte.gz")
    X_test = read_pixels("./data/t10k-images-idx3-ubyte.gz")
    y_test = read_labels("./data/t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test


# For Question 2.4
# Code to visualize weights (use your own weight variable, adjust its shape by yourself)
# plt.matshow(weight, cmap=plt.cm.gray, vmin=0.5*weight.min(), vmax=0.5*weight.max())
