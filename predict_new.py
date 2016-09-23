"""
Make predictions using pre-trained caffe model on new images.
Each image is first of all copied and augmented with lightening, darkening, rotation and reflection, 
and predictions are averaged across predictions for each augmented image.
 
:args: Mode - specify GPU (0) or CPU (1) mode for Caffe
:output: Text file containing image names and predicted class.
"""

import caffe
import argparse
import os
import glob
import re
import json
import numpy as np
import argparse
import skimage.exposure as exposure
import skimage.transform as transform
import predictions as predict


def predicted_class(probs):
    max_p = np.argmax(probs)
    return reverse_prediction_mapping[max_p]

DEPLOY = "model/deploy.prototxt"
MODEL = "model/_iter_5000.caffemodel"
MEAN = "model/alexnet_4_mean.npy"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mode", help="GPU or CPU mode. 0 for GPU, 1 for CPU")
    args = parser.parse_args()

    mode = args.mode

    if mode == 0:
        caffe.set_mode_gpu()
    elif mode == 1:
        caffe.set_mode_cpu()

    with open('data/class_mapping.json', 'r') as f:
        prediction_mapping = json.load(f)
    reverse_prediction_mapping = {v: k for k, v in prediction_mapping.items()}

    # Get image names
    image_list = glob.glob("data/images/*.jpg")

    # Initialize net and data transformer
    net, transformer = predict.initialize_model(DEPLOY, MODEL, MEAN)

    # Get predicted probabilities
    predicted_probs = predict.predict_images(image_list, net, transformer)
    # Save predictions to file
    predictions = [predicted_class(p) for p in predicted_probs]
    with open("data/images/predictions.txt", "w") as f:
        for i, c in zip(image_list, predictions):
            path, file = os.path.split(i)
            f.write("{} {}\n".format(file, c))
