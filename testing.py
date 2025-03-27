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

# Load in the MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Initialize model
convolutional = True
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
                adversarial_image, num_pixels_changed = create_adversarial_example_saliency_test(model, input, target_class,
                                                             epsilon, convolutional)

                # get prediction on adversarial image
                adv_pred, adv_probs = predict_sample(model, adversarial_image)
                #print(adv_pred, " versus ", target_class)
    
                if adv_pred == target_class:
                    success_matrix[source_class][target_class] += 1
                    print(f"Sample #{index} was a success")
                else:
                    print(f"Sample #{index} was a failure")
            # Percent of successes out of 25
            print(f"{source_class} was able to be forced to {target_class} successfully {success_matrix[source_class][target_class]*4}% of the time!")

# Generate visual
success_percent = (np.array(success_matrix) / 25.0) * 100
success_percent = success_percent.T

classes = [str(i) for i in range(10)]

plt.figure(figsize=(10, 8))
ax = sns.heatmap(success_percent, annot=False, fmt=".0f", cmap="YlOrRd", xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Success Rate (%)'})

# Labels and titles
plt.xlabel("Source Class")
plt.ylabel("Target Class")
plt.title("Adversarial Attack Success Rate (%)") #\n(25 attempts per pair)"
plt.tight_layout()
plt.show()

    


