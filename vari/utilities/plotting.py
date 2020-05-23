import numpy as np


def get_image_matrix(images):
    """Converts a list of [H,W,C] images to a single image matrix by concatenation"""
    rows = int(np.sqrt(len(images)))
    cols = rows
    image_rows = []
    for i in range(rows):
        image_rows += [np.concatenate(images[i*cols:(i+1)*cols], axis=1)]
    image_matrix = np.concatenate(image_rows, axis=0)
    return image_matrix


def plot_image_matrix(images, ax):
    """Plots a list of images in a grid in a sigle axis """
    image_matrix = get_image_matrix(images)
    ax.imshow(image_matrix, cmap='binary_r')
    ax.axis('off')
