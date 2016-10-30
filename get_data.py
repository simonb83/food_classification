"""
Download all required data for training and / or testing of model for classifying food images

Setting 'mode' flag to train will download:
- Resized images for relevant food classes (all as 256 x 256 pixels)
- Pre-trained Alexnet model weights

Setting 'mode' flag to test will download:
- Resized images for relevant food classes (all as 256 x 256 pixels)
- Fine-tuned Alexnet model weights
- Mean image binaryproto used during training

"""

import os
import sys
import argparse
import urllib
import time
import hashlib
import tarfile


def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = (time.time() - start_time) or 0.01
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def check_model(filename, sha1):
    """
    Check downloaded model vs provided sha1
    """
    with open(filename, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest() == sha1

# URL for downloading the model weights
model_urls = {
    'TRAIN': 'http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel',
    'TEST': 'https://s3-us-west-1.amazonaws.com/simon.bedford/food_classification/_iter_5000.caffemodel'
}

# For ensuring weights corretly downloaded
model_sha1 = {
    'TRAIN': '9116a64c0fbe4459d18f4bb6b56d647b63920377',
    'TEST': '78dbff81868c59e1afc1649d2e2007c53d646d05'
}

caffe_file = {
    'TRAIN': 'bvlc_alexnet.caffemodel',
    'TEST': '_iter_5000.caffemodel'
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="TRAIN or TEST mode. Setting flag to TRAIN will choose pre-trained model weights to be used in fine-tuning. Setting flag to TEST will choose fine-tuned model weights for testing purposes.")
    args = parser.parse_args()

    mode = args.mode

    # Download model
    filename = caffe_file[mode]
    model_filename = os.path.join('model/', filename)

    if os.path.exists(model_filename) and check_model(model_filename, model_sha1[mode]):
        print("Model already exists.")
        sys.exit(0)
        # Else download model
    else:
        urllib.urlretrieve(model_urls[mode], model_filename, reporthook)
        if not check_model(model_filename, model_sha1[mode]):
            print("Model did not download correctly. Try again.")
            sys.exit(1)

    # Download images
    images_url = "https://s3-us-west-1.amazonaws.com/simon.bedford/food_classification/resized.tar.gz"
    images_filename = os.path.join('data/', 'resized.tar.gz')
    images_sha1 = "adedce9915537851111c9b3a35df809568201753"

    if os.path.exists(images_filename) and check_model(images_filename, images_sha1):
        print("Images already downloaded.")
        sys.exit(0)
        # Else download images
    else:
        urllib.urlretrieve(images_url, images_filename, reporthook)
        if not check_model(images_filename, images_sha1):
            print("Images did not download correctly. Try again.")
            sys.exit(1)
        else:
            tar = tarfile.open(images_filename, "r:gz")
            tar.extractall()
            tar.close()

    # Download mean image
    mean_image_url = "https://s3-us-west-1.amazonaws.com/simon.bedford/food_classification/alexnet_4_mean.binaryproto"

    if mode == 'TEST':
        if os.path.exists('data/alexnet_4_mean.binaryproto'):
            print("Mean file already downloaded.")
            sys.exit(0)
        else:
            urllib.urlretrieve(
                mean_image_url, 'data/alexnet_4_mean.binaryproto', reporthook)
