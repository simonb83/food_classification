"""
Methods for use in testing and making new predictions
"""

import caffe
import argparse
import os
import numpy as np
import skimage.exposure as exposure
import skimage.transform as transform
import skimage.io as io


def standardize(image, size):
    """
    Standardize an image to a particular size and rotation
    :params image: image as numpy array
    :params size: standard image size as tuple
    :return: standardized image as numpy array
    """
    existing_shape = image.shape
    if existing_shape[1] > existing_shape[0]:
        image = transform.rotate(image, 90, resize=True)
    if image.shape != size:
        image = transform.resize(image, size)
    return image

def initialize_model(model_def, model_weights, mean_image):
    """
    Initialize a caffe model and data transformer
    :params model_def: Path to model prototxt file
    :params model_weights: Path to model weights
    :params mean_image: Path to mean image file
    :return net: Net object
    :return transformer: Transformer object
    """
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    mu = np.load(mean_image)
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    net.blobs['data'].reshape(1, 3, 227, 227)

    return net, transformer

def augment_image(img):
    """
    Augment an image using a combination of lightening, darkening, rotation and mirror images.
    :params img: Image as numpy array
    :return: array of augmented images 
    """
    augmented_images = []
    augmented_images.append(np.fliplr(img))
    for g in [0.45, 0.65, 0.85, 1.25, 1.5, 2]:
        new_img = exposure.adjust_gamma(img, gamma=g)
        augmented_images.append(new_img)
        augmented_images.append(np.fliplr(new_img))
    new_img = transform.rotate(img, 180)
    augmented_images.append(new_img)
    augmented_images.append(np.fliplr(new_img))
    return np.array(augmented_images)

def simple_predict(img, net, transformer):
    """
    Perform simple prediction on a single image by making a forward pass through the network
    :params img: Image as numpy array
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :return: array of predicted probabilities for each class
    """
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    p = net.forward()
    predictions = p['prob'][0].copy()
    return predictions

def predict_with_augment(img, net, transformer):
    """
    Perform predictions on original image plus a series of augmented images, and average across all predictions.
    :params img: Image as numpy array
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :return: array of predicted probabilities for each class
    """
    predictions = []
    predictions.append(simple_predict(img, net, transformer))
    aug_images = augment_image(img)
    for c in aug_images:
        predictions.append(simple_predict(c, net, transformer))
    predictions = np.array(predictions)
    return np.mean(predictions, axis=0)

def predict_images(image_list, net, transformer, p=''):
    """
    Iterate over a list of images and make predictions for each class based upon a specified prediction approach.
    :params image_list: Iterable of paths to images
    :params net: Initialized Net object
    :params transformer: Initialized Transformer object
    :params path: optional path to images
    :return: array of predicted class probabilities for each image of shape (num_images, num_classes)
    """
    predictions = []
    for i in image_list:
        path = os.path.join(p, i)
        img = caffe.io.load_image(path)
        # Resize if not the right shape
        shape = img.shape
        if shape != (256, 256, 3):
            img = standardize(img, (256, 256, 3))
        predictions.append(predict_with_augment(img, net, transformer))
    return np.array(predictions)