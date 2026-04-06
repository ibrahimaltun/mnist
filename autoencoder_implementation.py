import numpy as np

import keras
from keras import layers
from keras.datasets import mnist

from control_torch_for_gpu import set_gpu_to_model

print("keras vers: ", keras.__version__)

# #
set_gpu_to_model()

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
