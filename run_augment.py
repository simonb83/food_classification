"""
Augment a list of images by applying a combination of transforms to each image. Save each transformed image to disk,
and output a text file where each line contains the image path followed by the image class:
transformed_image_path image_class

:args images: Path to text file containing the list of images.
:args image_path: Path to root folder where images are currently saved.
:args image_output_path: Path to root directory where augmented images should be saved.
:args image_list_path: Path to root directoy for saving text file containing list of images and classes.
:output: Save each transformed image to disk, and save text file containing list of images and classes.
"""

import argparse
import os
import numpy as np
from skimage import io
import skimage.exposure as exposure
import skimage.transform as transform
import re


def save_image(img, image_name, operation):
    """
    Save image and its mirror image
    :params img: Image as numpy array
    :image_name: Name for base image
    :return: tuple of names of the two saved images
    """
    new_name = image_name + "_" + operation + ".jpg"
    mirror_name = image_name + "_" + operation + "m" + ".jpg"
    io.imsave(os.path.join("data/augmented", new_name), img)
    io.imsave(os.path.join(image_out, mirror_name), np.fliplr(img))
    return new_name, mirror_name


def augment_image(x):
    """
    Augment an image using a combination of lightening, darkening, rotation and mirror images, and save each image to disk.
    :params img: Image as numpy array
    :return: list of names of augmented images 
    """
    augmented_images = []

    pair = x.split(" ")
    label = " " + pair[1]
    im_path = pair[0]

    full_image_name = re.findall(r'\/(.+)', im_path)[0]
    image_name = re.findall(r'(\d+)\.', full_image_name)[0]

    # Load the image
    img = io.imread(os.path.join("data/augmented", im_path))

    # Save the original image
    n1, n2 = save_image(img, image_name, '')
    augmented_images.append(n1 + label)
    augmented_images.append(n2 + label)

    for i, g in enumerate([0.45, 0.65, 0.85, 1.25, 1.5, 2]):
        new_img = exposure.adjust_gamma(img, gamma=g)
        n1, n2 = save_image(new_img, image_name, "e{}".format(i + 1))
        augmented_images.append(n1 + label)
        augmented_images.append(n2 + label)

    new_img = transform.rotate(img, 180)
    n1, n2 = save_image(new_img, image_name, 'r')
    augmented_images.append(n1 + label)
    augmented_images.append(n2 + label)

    return augmented_images

if __name__ == "__main__":

    train_images = "data/train.txt"
    val_images = "data/val.txt"

    if not os.path.exists("data/augmented"):
        os.makedirs("data/augmented")

    with open(train_images, "r") as f:
        train_image_list = f.read().splitlines()

    with open(val_images, "r") as f:
        val_image_list = f.read().splitlines()

    new_train_image_list = []
    new_val_image_list = []

    for x in train_image_list:
        x_augmented = augment_image(x)
        for a in x_augmented:
            new_train_image_list.append(a)

    for x in val_image_list:
        x_augmented = augment_image(x)
        for a in x_augmented:
            new_val_image_list.append(a)

    np.random.shuffle(new_train_image_list)
    np.random.shuffle(new_val_image_list)

    with open("data/augmented_train.txt", "w") as f:
        f.write("\n".join(new_train_image_list))

    with open("data/augmented_val.txt", "w") as f:
        f.write("\n".join(new_val_image_list))
