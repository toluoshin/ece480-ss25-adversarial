# Testing script to evaluate effectiveness of adversarial attacks across different source/target pairs

# Import necessary modules
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from adversarial import create_adversarial_example_saliency, create_adversarial_example_saliency_test
from evaluate_model import evaluate_model, predict_sample
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import pickle
import os
import time

convolutional = False

if not convolutional and os.path.exists("mlp_success.pkl") and os.path.exists("mlp_distortion.pkl") and os.path.exists("mlp_time.pkl"):
    print("loading data!")
    with open("mlp_success.pkl", "rb") as f:
        success_matrix = pickle.load(f)
    with open("mlp_distortion.pkl", "rb") as f:
        pixels_matrix = pickle.load(f)
    with open("mlp_time.pkl", "rb") as f:
        time_matrix = pickle.load(f)
elif convolutional and os.path.exists("cnn_success.pkl") and os.path.exists("cnn_distortion.pkl") and os.path.exists("cnn_time.pkl"):
    print("loading data!")
    with open("cnn_success.pkl", "rb") as f:
        success_matrix = pickle.load(f)
    with open("cnn_distortion.pkl", "rb") as f:
        pixels_matrix = pickle.load(f)
    with open("cnn_time.pkl", "rb") as f:
        time_matrix = pickle.load(f)

else:
    # Load in the MNIST dataset
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Initialize model
    #convolutional = False
    print("Loading the trained model...")
    if convolutional:
        model = tf.keras.models.load_model('cnn_model.keras')
    else:
        model = tf.keras.models.load_model('mlp_model.keras')

    # Normalize and reshape data
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Data processing
    if convolutional:
        x_train = x_train.reshape((-1,28,28,1))
        x_test = x_test.reshape((-1,28,28,1))
    else:
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784)

    # # Initialize GUI
    root = tk.Tk()
    root.title("Adversarial Attacks")
    width= root.winfo_screenwidth()               
    height= root.winfo_screenheight()               
    root.geometry("%dx%d" % (width, height))

    # Initialize necessary variables
    success_matrix = [[0 for i in range(10)] for j in range(10)]
    pixels_matrix =  [[0 for i in range(10)] for j in range(10)]
    time_matrix =  [[0 for i in range(10)] for j in range(10)]
    digit_indices = [np.where(y_test == digit)[0] for digit in range(10)]
    epsilon = 1.0
    


    # test_indices = np.random.choice(digit_indices[0], size=25, replace=False)
    # for element in test_indices:
    #     print("Index ", element, " has classification ", y_test[element])

    # Loop to go through 25 combinations for each source/target pair
    for source_class in range(10):
        for target_class in range(10):
            if source_class == target_class:
                success_matrix[source_class][target_class] = 25
            else:
                # get 25 random samples from the loaded MNIST dataset
                test_indices = np.random.choice(digit_indices[source_class], size=25, replace=False)

                # for each input sample, test if attack was successful
                for index in test_indices:
                    input = np.expand_dims(x_test[index], axis=0)
                    #print(input.shape)
                    # adversarial_image, num_pixels_changed= create_adversarial_example_saliency(root, model, input, y_test[index],
                                                                        # target_class, epsilon, 0, convolutional)
                    start = time.time()
                    adversarial_image, num_pixels_changed = create_adversarial_example_saliency_test(model, input, target_class,
                                                                epsilon, convolutional)
                    end = time.time()

                    # get prediction on adversarial image
                    adv_pred, adv_probs = predict_sample(model, adversarial_image)
                    #print(adv_pred, " versus ", target_class)

                    if adv_pred == target_class:
                        success_matrix[source_class][target_class] += 1
                        pixels_matrix[source_class][target_class] += num_pixels_changed
                        time_matrix[source_class][target_class] = end-start
                        print(f"Sample #{index} was a success")
                    else:
                        print(f"Sample #{index} was a failure")
                # Percent of successes out of 25
                print(f"{source_class} was able to be forced to {target_class} successfully {success_matrix[source_class][target_class]*4}% of the time!")
    # calculate distortion percentage
    # Initialize result matrix with zeros
    success_matrix = np.array(success_matrix, dtype=float)
    pixels_matrix = np.array(pixels_matrix, dtype=float)
    time_matrix = np.array(time_matrix, dtype=float)

    if not convolutional:
        with open("mlp_success.pkl", "wb") as f:
            pickle.dump(success_matrix, f)
        with open("mlp_distortion.pkl", "wb") as f:
            pickle.dump(pixels_matrix, f)
        with open("mlp_time.pkl", "wb") as f:
            pickle.dump(time_matrix, f)
    else:
        with open("cnn_success.pkl", "wb") as f:
            pickle.dump(success_matrix, f)
        with open("cnn_distortion.pkl", "wb") as f:
            pickle.dump(pixels_matrix, f)
        with open("cnn_time.pkl", "wb") as f:
            pickle.dump(time_matrix, f)


success_percent = (np.array(success_matrix) / 25.0) * 100
success_percent = success_percent.T

# # Create a mask where success_matrix is not zero
# mask = success_matrix != 0

# distortion_percent = np.zeros_like(pixels_matrix, dtype=float)
# distortion_percent[mask] = (pixels_matrix[mask] / success_matrix[mask]) / 784
# time_difference = np.zeros_like(time_matrix, dtype=float)
# time_difference[mask] = (time_matrix[mask] / success_matrix[mask])
average_time = np.sum(time_matrix) / np.sum(success_matrix)
average_distortion = (np.sum(pixels_matrix) / np.sum(success_matrix)) / 784

# Print stats:
print("Average time for attacks was ", average_time)
print("Average distortion for attacks was ", average_distortion)
print("Attack success rate was ", np.sum(success_matrix)/2500)

# # Generate visual
success_percent = (np.array(success_matrix) / 25.0) * 100
success_percent = success_percent.T




classes = [str(i) for i in range(10)]

plt.figure(figsize=(10, 8))
ax = sns.heatmap(success_percent, annot=False, fmt=".0f", cmap="YlOrRd", xticklabels=classes, yticklabels=classes, vmin=0, vmax=100, cbar_kws={'label': 'Success Rate (%)'})
ax.invert_yaxis()
cbar = ax.collections[0].colorbar
cbar.set_label('Success Rate (%)', fontsize=30)
cbar.ax.tick_params(labelsize=20)
# Force color bar range from 0 to 100
#ax.collections[0].colorbar.set_clim(0, 100)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=30)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=30)
ax.set_xlabel("Source Class", fontsize=30)
ax.set_ylabel("Target Class", fontsize=30)

# Labels and titles
# plt.xlabel("Source Class")
# plt.ylabel("Target Class")
if convolutional:
    #plt.title("CNN Adversarial Attack Success Rate (%)") #\n(25 attempts per pair)"
    ax.set_title("CNN Adversarial Attack Success Rate", fontsize=30, pad=15)
else:
    # plt.title("MLP Adversarial Attack Success Rate (%)") #\n(25 attempts per pair)"
    ax.set_title("MLP Adversarial Attack Success Rate", fontsize=30, pad=15)
plt.tight_layout()
plt.show()



    


