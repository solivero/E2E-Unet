import tensorflow as tf
from UNet_MSOF_model import Nest_Net2

import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow_io as tfio

# For more information about autotune:
# https://www.tensorflow.org/guide/data_performance#prefetching
AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

root = "/app"
dataset_path = os.path.join(root, "spacenet7")
training_data = "train/"
val_data = "train/"
# Image size that we are going to use
IMG_SIZE = 1024
SEED = 42

def parse_image_pair(img1_path: str, img2_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    image1 = tf.io.read_file(img1_path)
    image1 = tfio.experimental.image.decode_tiff(image1)
    image1 = tf.image.convert_image_dtype(image1, tf.uint8)[:, :, :3]

    image2 = tf.io.read_file(img2_path)
    image2 = tfio.experimental.image.decode_tiff(image2)
    image2 = tf.image.convert_image_dtype(image2, tf.uint8)[:, :, :3]

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    mask_path = tf.strings.regex_replace(img1_path, "images_masked", "change_maps")
    prefix_len = len('global_monthly_')
    date_len = 7
    img1_file = tf.strings.split(img1_path, sep='/')[-1]
    date1 = tf.strings.substr(img1_file, prefix_len, date_len)
    img2_file = tf.strings.split(img2_path, sep='/')[-1]
    date2 = tf.strings.substr(img2_file, prefix_len, date_len)
    double_date = tf.strings.join([date1, date2], separator='-')

    cm_name = tf.strings.regex_replace(mask_path, r'20\d{2}_\d{2}', double_date)

    #cm_name = mask_path

    mask = tf.io.read_file(cm_name)
    # The masks contain a class index for each pixels
    mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)[:, :, :1]
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
    filler_row = tf.zeros((1, IMG_SIZE, 1), tf.uint8)
    mask = tf.concat([mask, filler_row], axis=0)

    # Note that we have to convert the new value (0)

    merged_image = tf.concat([image1, image2], axis=2)
    filler_row = tf.zeros((1, IMG_SIZE, 6), tf.uint8)
    merged_image = tf.concat([merged_image, filler_row], axis=0)

    return {'image': merged_image, 'segmentation_mask': mask}
    # return merged_image, mask


#val_dataset = tf.data.Dataset.list_files(dataset_path + val_data + "*.tif", seed=SEED)
#val_dataset = val_dataset.map(parse_image)
# Here we are using the decorator @tf.function
# if you want to know more about it:
# https://www.tensorflow.org/api_docs/python/tf/function

@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between 0.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint: dict) -> tuple:
    """Normalize and resize a test image and its annotation.

    Notes
    -----
    Since this is for the test set, we don't need to apply
    any data augmentation technique.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """
    input_image = tf.image.resize(datapoint['image'], (IMG_SIZE, IMG_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (IMG_SIZE, IMG_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

train_dataset = tf.data.Dataset.list_files(os.path.join(dataset_path, training_data + "L15-0357E-1223N_1429_3296_13/images_masked/*.tif"), shuffle=False)
train_image_pairs = tf.data.Dataset.zip((train_dataset, train_dataset.skip(1)))
train_dataset = train_image_pairs.map(parse_image_pair)
item = next(iter(train_dataset))
print(item['image'].shape)
print(item['segmentation_mask'].shape)
input_shape = [1024, 1024, 6]
# for reference about the BUFFER_SIZE in shuffle:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BUFFER_SIZE = 1000
BATCH_SIZE = 6

dataset = {"train": train_dataset, "val": train_dataset}

# -- Train Dataset --#
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=SEED)
#dataset['train'] = dataset['train'].repeat(2)
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)

Nest_Net2(input_shape).fit(dataset['train'])