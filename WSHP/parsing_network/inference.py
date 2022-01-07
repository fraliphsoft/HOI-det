"""Run DeepLab-ResNet on a given image.

This script computes a segmentation mask for a given image.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

from PIL import Image

import tensorflow as tf
tf = tf.compat.v1
tf.disable_v2_behavior()
import numpy as np
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label

import pdb

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

dataset_name = 'hico'  # 'hico' or 'vcoco'
phrase = 'train'  # 'train' or 'test'

NUM_CLASSES = 7
MODEL = './models/final_model/model.ckpt-19315'
DATA_LIST = f'./filename_hico_train.txt'
# SAVE_DIR = f"output/"
SAVE_DIR = f"../../data/{dataset_name}/humans/{phrase+'2015'}/"

# setting for hico-test
DATASET_PATH = f"../../data/hico/images/{phrase+'2015'}/"
# setting for vcoco-test
# DATASET_PATH = f"../../data/vcoco/images/{phrase+'val'}/"
# DATASET_PATH = f"../../data/vcoco/images/{phrase}/"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network Inference.")
    parser.add_argument("--img_path", type=str,
                        help="Path to the RGB image file folder.", default=DATASET_PATH)
    parser.add_argument("--model_weights", type=str,
                        help="Path to the file with model weights.", default=MODEL)
    parser.add_argument("--data_list", type=str, default=DATA_LIST,
                        help="Path to the image list.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Where to save predicted mask.")
    return parser.parse_args()


def load(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    num_steps = file_len(args.data_list)
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.img_path,
            args.data_list,
            None,  # No defined input size.
            False,  # No random scale.
            False,  # No random mirror.
            255,
            IMG_MEAN,
            coord)
        image, label = reader.image, reader.label
        title = reader.queue[0]
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0)  # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()

    # Predictions.
    raw_output = net.layers['fc1_voc12']

    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3, ])
    print(raw_output_up.shape)

    # raw_output_up = tf.argmax(raw_output_up, dimension=3)
    # print(raw_output_up.shape)

    # pred = tf.expand_dims(raw_output_up, dim=3)
    # print(pred.shape)

    # Set up TF session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    load(loader, sess, args.model_weights)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    start_time = time.time()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Perform inference.-
    for step in tqdm(range(num_steps)):
    # for step in tqdm(range(10)):
        preds, jpg_path = sess.run([raw_output_up, title])
        preds = preds[0]
        print(preds.shape)
        print('Image processed {}'.format(jpg_path))

        name1 = str(jpg_path).split('.')[-2]
        name2 = name1.split('/')[-1]

        np.save(args.save_dir + name2 + '.npy', preds)

    total_time = time.time() - start_time
    print('The output files have been saved to {}'.format(args.save_dir))
    print('It took {} sec on each image.'.format(total_time / num_steps))


if __name__ == '__main__':
    main()
