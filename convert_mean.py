"""
Convert an image mean in binaryproto format to numpy array

:args mean_file: Path to mean image binaryproto file
:output: Saves converted file to the same folder as .npy file
"""


import caffe
import argparse
import os
import numpy as np
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='make predictions')
    parser.add_argument("-m", "--mean_file", type=str, nargs=1,
                        help='Path to mean image binary proto')

    args = parser.parse_args()
    mean_file = args.mean_file[0]

    blob = caffe.proto.caffe_pb2.BlobProto()
    with open(mean_file, "rb") as f:
        data = f.read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))

    file_name = re.findall(r'\/(\w+)\.binaryproto', mean_file)[0]
    file_path = re.findall(r'(.+)\/\w+.binaryproto', mean_file)[0]

    np.save(os.path.join(file_path, file_name + ".npy"), arr[0])
