import matplotlib.pyplot as plt
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

# # 2. AUTOENCODER MODEL
enconding_dim = 32  # represents the size of encoded

input_img = keras.Input(shape=(784,))  # input image as array like 28x28

# "encoded" is the encoded representation of the input
encoded = layers.Dense(enconding_dim, activation='relu')(input_img)

# 'decoded' is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# This model maps as input its encoded representation
encoder = keras.Model(input_img, encoded)

# Decoder Model
# This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(enconding_dim,))

# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]

# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

# Compile the model
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

# # 3. TRAIN AUTOENCODER

autoencoder.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    # verbose=1,
    validation_data=(x_test, x_test)
)

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
