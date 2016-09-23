# Machine Learning & Food Classification

This repository contains code for fine-tuning a Convolutional Neural Network in order to classify a set of food images. It is based on all the work carried out for the Capstone project for Springboard's Data Science Intensive course. 

## Dependencies:

Dependencies can be installed with:

~~~~
pip install requirements.txt
~~~~

Caffe needs to be compiled (from master branch), including PYTHON wrapper `make pycaffe`

## Instructions for use:

1. Clone the repo 

~~~~
git clone https://github.com/simonb83/food_classification
~~~~

2. cd into the food_classification folder

3. Get the data and relevant model weights. 
 
If you wish to run both training and testing, use flag -m TRAIN:

~~~~
python get_data.py -m TRAIN
~~~~

If you only wish to run testing code, use flag -m TEST:

~~~~
python get_data.py -m TEST
~~~~

#### Training:

1. Run data augmentation:

~~~~
python run_augment.py
~~~~

2. Create lmdb databases for caffe

~~~~
./path-to-caffe-root/build/tools/convert_imageset  \
data/augmented/ \
data/augmented_train.txt \
data/train_lmdb

./path-to-caffe-root/build/tools/convert_imageset  \
data/augmented/ \
data/augmented_val.txt \
data/val_lmdb
~~~~

3. Generate mean image of training data

~~~~
./path-to-caffe-root/build/tools/compute_image_mean \
-backend=lmdb \
data/train_lmdb \
data/alexnet_4_mean.binaryproto
~~~~

4. Update paths in train_val.prototxt

Update path to data_classification directory in mean_file and source lines:

~~~~
mean_file: "~your_path/food_classification/data/alexnet_4_mean.binaryproto"
source: "~your_path/food_classification/data/train_lmdb"

source: "~your_path/food_classification/data/val_lmdb"
~~~~

5. Update paths in solver.prototxt

Update path to data_classification directory in net and snapshot_prefix lines:

~~~~
net: "~your_path/food_classification/model/train_val.prototxt"
snapshot_prefix: "~your_path/food_classification/model/"
~~~~

6. (Optional) Change flag from GPU to CPU in solver.prototxt
~~~~
solver_mode: GPU
~~~~

7. Train the model

~~~~
./path-to-caffe-root/build/tools/caffe train \
-solver /food_classification/model/solver.prototxt \
-weights /food_classification/model/bvlc_alexnet.caffemodel \
\> /food_classification/data/train.log
~~~~


#### Testing:

1. Convert mean image

~~~~
python convert_mean.py -m data/alexnet_4_mean.binaryproto
~~~~

2. Make predictions

~~~~
python run_predict.py
~~~~

To visualize predictions, use the visualize.ipynb notebook.


#### Making New Predictions:

The model can also be used for making predictions on new images.

Requirements:

1. If you have not already done so, download data with TEST flag:

~~~~
python get_data.py -m TEST
~~~~

2. Convert mean image:

~~~~
python convert_mean.py -m data/alexnet_4_mean.binaryproto
~~~~

3. Add some images to the `'data/images'` directory; the expected image format is jpg.

4. From the root directory run the prediction script, with `-m 0` for GPU, `-m -1` for CPU:

~~~~
python predict_new.py -m 0
~~~~

5. The predictions will be output to `'data/images/predictions.txt'`

See more [here](../data/images/readme.md)

