import numpy as np

import tensorflow as tf
import keras
from keras import layers
from keras.datasets import mnist

print("keras vers: ", keras.__version__)

# # 0. Set GPU device to use
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(
            memory_limit=8192
        )]
    )
    print("It will use 8GB VRAM to train the model")


# # 1. DATA PREPARATION
# mnist data images has 28x28 dimension
(x_train, _), (x_test, _) = mnist.load_data()
print("train data shape: ", x_train.shape)
print("test data shape: ", x_test.shape)

# Normalize all values between 0 and 1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("normalized train data shape", x_train.shape)
print("normalized test data shape", x_test.shape)
