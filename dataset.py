import tensorflow as tf

import numpy as np
import tensorflow as tf
import datetime
import os
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

root = "/app"
dataset_path = os.path.join(root, "spacenet7")
training_data = "train/"
val_data = "train/"
IMG_SIZE = 1024 # Image size expected on file
UPSCALE = 2 # To avoid tiny areas of 2x2 pixels
PATCH_SIZE = 512 # After upscaling
SEED = 42

def parse_image_pair(csv_batch) -> dict:
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
    img1_path = csv_batch['image1'][0]
    image1 = tf.io.read_file(img1_path)
    image1 = tf.image.decode_png(image1, channels=3)
    image1 = tf.image.convert_image_dtype(image1, tf.uint8)[:, :, :3]

    img2_path = csv_batch['image2'][0]
    image2 = tf.io.read_file(img2_path)
    image2 = tf.image.decode_png(image2, channels=3)
    image2 = tf.image.convert_image_dtype(image2, tf.uint8)[:, :, :3]

    #cm_name = tf.strings.regex_replace(mask_path, r'20\d{2}_\d{2}', double_date)
    cm_name = csv_batch['label'][0]

    #cm_name = mask_path

    mask = tf.io.read_file(cm_name)
    # The masks contain a class index for each pixels
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)[:, :, :1]
    mask = tf.where(mask == 255, np.dtype('uint8').type(1), mask)
    #filler_row = tf.zeros((1, 1024, 1), tf.uint8)
    #mask = tf.concat([mask, filler_row], axis=0)

    # Note that we have to convert the new value (0)

    merged_image = tf.concat([image1, image2], axis=2)
    #filler_row = tf.zeros((1, 1024, 6), tf.uint8)
    #merged_image = tf.concat([merged_image, filler_row], axis=0)

    #return {'image': merged_image, 'segmentation_mask': mask}
    return merged_image, mask

@tf.function
def make_patches(image: tf.Tensor, mask: tf.Tensor):
    n_patches = ((IMG_SIZE*UPSCALE) // PATCH_SIZE)**2
    image_patches = tf.image.extract_patches(images=tf.expand_dims(image, 0),
                        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')[0]
    image_patch_batch = tf.reshape(image_patches, (n_patches, PATCH_SIZE, PATCH_SIZE, 6))
    mask_patches = tf.image.extract_patches(images=tf.expand_dims(mask, 0),
                        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
                        rates=[1, 1, 1, 1],
                        padding='SAME')[0]
    mask_patch_batch = tf.reshape(mask_patches, (n_patches, PATCH_SIZE, PATCH_SIZE, 1))
    return image_patch_batch, mask_patch_batch


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
def upscale_images(image: tf.Tensor, mask: tf.Tensor) -> tuple:
    upscaled_size = IMG_SIZE*UPSCALE
    # use nearest neightbor?
    input_image = tf.image.resize(image, (upscaled_size, upscaled_size))
    input_mask = tf.image.resize(mask, (upscaled_size, upscaled_size))
    return input_image, input_mask

@tf.function
def downscale_images(image: tf.Tensor, mask: tf.Tensor) -> tuple:
    downscaled_size = IMG_SIZE//UPSCALE
    # use nearest neightbor?
    input_image = tf.image.resize(image, (downscaled_size, downscaled_size))
    input_mask = tf.image.resize(mask, (downscaled_size, downscaled_size))
    return input_image, input_mask

@tf.function
def load_image_train(image: tf.Tensor, mask: tf.Tensor) -> tuple:
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

    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    input_image, input_mask = normalize(image, mask)

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
    input_image = tf.image.resize(datapoint['image'], (PATCH_SIZE, PATCH_SIZE))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (PATCH_SIZE, PATCH_SIZE))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

def load_csv_dataset(csv_path):
    return tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=1, # Actual batching in later stages
        num_epochs=1,
        ignore_errors=True)
        # Shuffle train_csv_ds first to have diverse val set?

def make_patches_ds(image, mask):
    return tf.data.Dataset.from_tensor_slices(make_patches(image, mask))

def load_image_dataset(csv_dataset):
    return csv_dataset \
        .map(parse_image_pair) \
        .map(upscale_images) \
        .flat_map(make_patches_ds) \
        .map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE) \

def load_datasets(csv_path, batch_size=8, val_size=128, buffer_size=100):
    csv_dataset = load_csv_dataset(csv_path)
    train_csv = csv_dataset.skip(val_size)
    dataset_train = load_image_dataset(train_csv) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE) \
        .shuffle(buffer_size=buffer_size, seed=SEED)
    val_csv = csv_dataset.take(val_size)
    dataset_val = load_image_dataset(val_csv) \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=AUTOTUNE)
    return dataset_train, dataset_val
