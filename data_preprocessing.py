from keras.datasets import mnist
import numpy as np

# Load the MNIST dataset (train and test sets)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize input images to the range [0, 1]
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# Reshape the data to fit the Theano input format (batch_size, channels, height, width)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

# Convert labels to one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

print(f'Train data shape: {x_train.shape}, Train labels shape: {y_train.shape}')
print(f'Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}')
