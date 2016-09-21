"""
Make predictions using pre-trained caffe model, first of all augmenting test images using same methodology as used for augmenting training data,
and averaging across predictions for each augmented test image.

:output: Save per class predictions for each test image and prediction method.
"""

import caffe
import argparse
import os
import numpy as np
import skimage.exposure as exposure
import skimage.transform as transform
import skimage.io as io
import predictions as predict

DEPLOY = "model/deploy.prototxt"
MODEL = "model/_iter_5000.caffemodel"
MEAN = "data/alexnet_4_mean.npy"
caffe.set_mode_gpu()

image_list = []
with open("/data/test.txt", "r") as f:
    test_images = f.read().splitlines()

for im in test_images:
    image_list.append(im.split(" ")[0])

#Initialize net and data transformer
net, transformer = predict.initialize_model(DEPLOY, "model/_iter_5000.caffemodel", "data/alexnet_4_mean.npy")

#Make predictions
predictions = predict.predict_images(image_list, net, transformer, p='data/resized')

#Dump predictions
predictions.dump("data/predictions")
