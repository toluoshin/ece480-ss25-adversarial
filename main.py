import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk, ImageChops
import io
from train_model import train_model
from evaluate_model import evaluate_model, predict_sample
from adversarial import create_adversarial_example, create_adversarial_example_saliency
from drawing import get_drawn_digit
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 


# def get_sample_by_digit(x_test, y_test, source_digit):
#     """
#     Get a sample image of the specified digit from the test set.

#     Parameters:
#     x_test (numpy.ndarray): The test images dataset
#     y_test (numpy.ndarray): The test labels dataset
#     source_digit (int): The digit to find (0-9)

#     Returns:
#     tuple: (sample_image, sample_label) where sample_image is the first instance
#            of the target digit found in the test set
#     """
#     # Find all indices where the label matches the target digit
#     digit_indices = np.where(y_test == source_digit)[0]

#     if len(digit_indices) == 0:
#         raise ValueError(f"No instances of digit {source_digit} found")

#     # Select random index instead of first one
#     sample_idx = np.random.choice(digit_indices)
#     return x_test[sample_idx:sample_idx + 1], y_test[sample_idx:sample_idx + 1]


def preprocess_uploaded_image(image_path, convolutional):
    """
    Preprocess an uploaded image to match MNIST format.
    """
    try:
        # Load and convert image to grayscale
        img = Image.open(image_path).convert('L')

        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0

        # Invert if necessary (MNIST has white digits on black background)
        if img_array.mean() > 0.5:
            img_array = 1 - img_array

        # Flatten the array for model input
        if convolutional:
            img_array.reshape((28,28))
            img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
        else:
            img_array = img_array.reshape(1, 784)

        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

'''def upload_image_with_base(base_image_path,new_image_path):
    try:
        # Load and convert image to grayscale
        img_b = Image.open(base_image_path).convert('L')
        img_n = Image.open(new_image_path).convert('L')

        # Resize to 28x28
        img_b = img_b.resize((28, 28), Image.Resampling.LANCZOS)
        img_n = img_n.resize((28, 28), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize
        img_array_b = np.array(img_b, dtype=np.float32)
        img_array_b = img_array_b / 255.0
        img_array_n = np.array(img_n, dtype=np.float32)
        img_array_n = img_array_n / 255.0

        # Invert if necessary (MNIST has white digits on black background)
        if img_array_b.mean() > 0.5:
            img_array_b = 1 - img_array_b
        if img_array_n.mean() > 0.5:
            img_array_n = 1 - img_array_n

        # Subtract pixels using Matrix/Array subtraction
        result_array = np.subtract(img_array_n, img_array_b)

        # Flatten the array for model input
        result_array = result_array.reshape(1, 784)

        return img_array_n
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")'''

def plot_uploaded_comparison(uploaded_image, adversarial_image=None):
    """Plot the uploaded image and its adversarial version if available."""
    if tf.is_tensor(uploaded_image):
        uploaded_image = uploaded_image.numpy()

    # Create subplots based on whether adversarial image is provided
    n_plots = 3 if adversarial_image is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]  # Make axes iterable for single plot

    # Plot uploaded image
    uploaded_display = uploaded_image.reshape(28, 28)
    im1 = axes[0].imshow(uploaded_display, cmap='gray')
    axes[0].set_title("Uploaded Image")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    if adversarial_image is not None:
        if tf.is_tensor(adversarial_image):
            adversarial_image = adversarial_image.numpy()

        # Plot adversarial image
        adversarial_display = adversarial_image.reshape(28, 28)
        im2 = axes[1].imshow(adversarial_display, cmap='gray')
        axes[1].set_title("Adversarial Image")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # Plot difference
        difference = adversarial_image - uploaded_image
        difference_display = difference.reshape(28, 28)
        max_diff = max(abs(difference_display.max()), abs(difference_display.min()))
        im3 = axes[2].imshow(difference_display, cmap='seismic', vmin=-max_diff, vmax=max_diff)
        axes[2].set_title("Perturbation (Difference)")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()


# def plot_images(root, original, adversarial, difference, adv_probs):
#     """Plot the original image, adversarial image, and their difference."""
#     # Convert to numpy if tensor
#     if tf.is_tensor(original):
#         original = original.numpy()
#     if tf.is_tensor(adversarial):
#         adversarial = adversarial.numpy()
#     if tf.is_tensor(difference):
#         difference = difference.numpy()

#     # Reshape images for display
#     original = original.reshape(28, 28)
#     adversarial = adversarial.reshape(28, 28)
#     difference = difference.reshape(28, 28)

#     clear_screen(root)
#     # menu button
#     # menu_btn = ttk.Button(root, text="Back to Menu", command=lambda: load_menu())
#     # menu_btn.pack(pady = 50)

#     fig = Figure(figsize=(5,5))
#     axes = fig.subplots(2,2)
#     canvas = FigureCanvasTkAgg(fig, master = root)
#     canvas.draw()
#     canvas.get_tk_widget().pack(anchor="center", expand=True, fill="both")

#     #fig, axes = plt.subplots(1, 4, figsize=(15, 5))

#     # Plot original image
#     im1 = axes[0,0].imshow(original, cmap='gray')
#     axes[0,0].set_title("Original Image")
#     axes[0,0].axis('off')
#     fig.colorbar(im1, ax=axes[0,0])

#     # Plot adversarial image
#     im2 = axes[0,1].imshow(adversarial, cmap='gray')
#     axes[0,1].set_title("Adversarial Image")
#     axes[0,1].axis('off')
#     fig.colorbar(im2, ax=axes[0,1])

#     # Plot the difference (noise)
#     # Normalize difference to symmetric range for seismic colormap
#     max_diff = max(abs(difference.max()), abs(difference.min()))
#     im3 = axes[1,0].imshow(difference, cmap='seismic', vmin=-max_diff, vmax=max_diff)
#     axes[1,0].set_title("Perturbation (Difference)")
#     axes[1,0].axis('off')
#     fig.colorbar(im3, ax=axes[1,0])

#     axes[1,1].set_title("Confidence Scores")
#     digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     bars = axes[1,1].bar(digits, adv_probs)

#     fig.tight_layout()
#     # plt.tight_layout()
#     # plt.show()

def plot_confidence_text(min_epsilon_predictions, min_topn_predictions, min_distortion_predictions):
    """Display confidence rates in a separate window"""
    plt.figure(num=3, figsize=(12, 12))
    plt.ylim(0, 1)
    plt.text(0.05, 0.95, "Top 3 Confidence Rates for Min Epsilon:", fontsize=12, fontweight='bold', va='top')

    y_pos = 0.8
    for i, (label, conf) in enumerate(min_epsilon_predictions, 1):
        plt.text(0.1, y_pos, f"{i}: Class {label} - {conf * 100:.2f}%", fontsize=11, va='top')
        y_pos -= 0.05

    y_pos -= 0.1  # Add some space between sections
    plt.text(0.05, y_pos, "Top 3 Confidence Rates for Min Top N:", fontsize=12, fontweight='bold', va='top')

    y_pos -= 0.05
    for i, (label, conf) in enumerate(min_topn_predictions, 1):
        plt.text(0.1, y_pos, f"{i}: Class {label} - {conf * 100:.2f}%", fontsize=11, va='top')
        y_pos -= 0.05

    y_pos -= 0.1  # Add some space between sections
    plt.text(0.05, y_pos, "Top 3 Confidence Rates for Min Distortion:", fontsize=12, fontweight='bold', va='top')

    y_pos -= 0.05
    for i, (label, conf) in enumerate(min_distortion_predictions, 1):
        plt.text(0.1, y_pos, f"{i}: Class {label} - {conf * 100:.2f}%", fontsize=11, va='top')
        y_pos -= 0.05

    plt.axis('off')  # Hide axes
    plt.show(block=False)

def plot_success_rate(success_rates, root):
    # Convert success rates to a format that can be plotted
    epsilon_values = np.arange(0.1, 1.1, 0.1)
    top_n_values = range(0, 201, 5)

    success_matrix = np.zeros((len(top_n_values), len(epsilon_values)))

    for epsilon, top_n, distortion, success, adv_probs, adv_image in success_rates:
        epsilon_idx = int((epsilon-0.1) * 10)
        top_n_idx = (top_n) // 5
        success_matrix[top_n_idx, epsilon_idx] += success

    # Normalize the success rate matrix
    if np.max(success_matrix) > 0:
        success_matrix = success_matrix / np.max(success_matrix)
    else:
        success_matrix = np.zeros_like(success_matrix)

    # Create a figure
    #plt.clf()  
    #fig, ax = plt.subplots(figsize=(10, 6))
    fig = Figure(figsize=(10,6))
    ax = fig.subplots(1,1)
    cax = ax.imshow(success_matrix, cmap='YlGnBu', aspect='auto', origin='lower')

    fig.colorbar(cax, label="Success Rate")
    ax.set_xticks(np.arange(len(epsilon_values)))
    ax.set_xticklabels([f'{epsilon:.1f}' for epsilon in epsilon_values])
    ax.set_yticks(np.arange(len(top_n_values)))
    ax.set_yticklabels([str(top_n) for top_n in top_n_values])
    ax.set_xlabel("Epsilon Value")
    ax.set_ylabel("Top N Pixels Perturbed")
    ax.set_title("Success Rate of Adversarial Attack Using Saliency Map")
    fig.tight_layout()

    # Embed the plot in the tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_min_success_cases(success_rates, model, original_image, original_label, target_label, root):
    """
    Plot two cases:
    1. Minimum epsilon required for a successful adversarial attack
    2. Minimum number of pixels changed required for a successful adversarial attack
    """
    # Case 1: Find the minimum epsilon and corresponding pixel count that resulted in a successful attack
    min_epsilon_case = min(
        ((epsilon, top_n, distortion, success, adv_probs, adversarial_image) for epsilon, top_n, distortion, success, adv_probs, adversarial_image in success_rates if success == 1),
        key=lambda x: x[0],
        default=None
    )

    # Case 2: Find the minimum number of pixels changed and corresponding epsilon that resulted in a successful attack
    min_top_n_case = min(
        ((epsilon, top_n, distortion, success, adv_probs, adversarial_image) for epsilon, top_n, distortion, success, adv_probs, adversarial_image in success_rates if success == 1),
        key=lambda x: x[1],
        default=None
    )

    # Case 3: Find the minimum distortion that resulted in a successful attack
    min_distortion_case = min(
        ((epsilon, top_n, distortion, success, adv_probs, adversarial_image) for epsilon, top_n, distortion, success, adv_probs, adversarial_image in success_rates if success == 1),
        key=lambda x: x[2],
        default=None
    )

    # Check if we have successful cases
    display_frame = Frame(root, background="white", relief="sunken")

    if min_epsilon_case is None or min_top_n_case is None:
        print("No successful adversarial examples found.")
        #display_frame = Frame(root, background="white", relief="sunken")
        # failure_label = Label(display_frame, text="No successful adversarial examples found.", bg="white", fg="black", font=("Arial", 25, "bold"))
        # failure_label.pack(pady=2, fill="both", expand=True)
        # failure_label.pack()
        fig = Figure(figsize=(9,6.5))
        fig.suptitle("No successful adversarial examples found.", fontsize=20, y=0.5)
        canvas = FigureCanvasTkAgg(fig, master=display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both")
    else:
        #display_frame.pack(fill="both", expand=True)

        #print("min_epsilon_case: ", min_epsilon_case)
        #print("min_top_n_case: ", min_top_n_case)

        # Extract epsilon and top_n values for each case
        min_epsilon, min_epsilon_top_n, min_epsilon_distortion, min_epsilon_success, min_epsilon_confidence, min_epsilon_adv_image = min_epsilon_case
        min_top_n_epsilon, min_top_n, min_top_n_distortion, min_top_n_success, min_top_n_confidence, min_top_n_adv_image = min_top_n_case
        min_distortion_epsilon, min_distortion_top_n, min_distortion, min_distortion_success, min_distortion_confidence, min_distortion_adv_image = min_distortion_case

        # # Generate adversarial images for the two cases
        # min_epsilon_adv_image, _ = create_adversarial_example_saliency(model, original_image, original_label, target_label, min_epsilon,
        #                                                             min_epsilon_top_n)
        # min_top_n_adv_image, _ = create_adversarial_example_saliency(model, original_image, original_label, target_label, min_top_n_epsilon,
        #                                                           min_top_n)

        # Convert tensors to numpy arrays for plotting
        original_image_np = original_image.numpy() if tf.is_tensor(original_image) else original_image
        min_epsilon_adv_image_np = min_epsilon_adv_image.numpy() if tf.is_tensor(
            min_epsilon_adv_image) else min_epsilon_adv_image
        min_top_n_adv_image_np = min_top_n_adv_image.numpy() if tf.is_tensor(min_top_n_adv_image) else min_top_n_adv_image
        min_distortion_adv_image_np = min_distortion_adv_image.numpy() if tf.is_tensor(min_distortion_adv_image) else min_distortion_adv_image

        # Calculate perturbations
        #if np.allclose(min_epsilon_adv_image_np, original_image_np, atol=1e-6)
        perturbation_min_epsilon = np.abs(min_epsilon_adv_image_np - original_image_np)
        perturbation_min_top_n = np.abs(min_top_n_adv_image_np - original_image_np)
        perturbation_min_distortion = np.abs(min_distortion_adv_image_np - original_image_np)

        # Calculate confidence for each adversarial image (top 3 confidences)
        # min_epsilon_confidence = model.predict(min_epsilon_adv_image_np.reshape(1, 784))[0]
        # min_top_n_confidence = model.predict(min_top_n_adv_image_np.reshape(1, 784))[0]
        # min_epsilon_confidence = predict_sample(model, min_epsilon_adv_image_np)[1]
        # min_top_n_confidence = predict_sample(model, min_top_n_adv_image_np)[1]
        # min_epsilon_confidence = success_rates
        # min_top_n_confidence = predict_sample(model, min_top_n_adv_image_np)[1]

        # print("min_epsilon_confidence: ", min_epsilon_confidence)
        # print("min_top_n_confidence: ", min_top_n_confidence)

        # Get the top 3 confidence values and their corresponding classes
        min_epsilon_top_3 = np.argsort(min_epsilon_confidence)[::-1][:3]
        min_top_n_top_3 = np.argsort(min_top_n_confidence)[::-1][:3]
        min_distortion_top_3 = np.argsort(min_distortion_confidence)[::-1][:3]

        min_epsilon_confidence_top_3 = min_epsilon_confidence[min_epsilon_top_3]
        min_top_n_confidence_top_3 = min_top_n_confidence[min_top_n_top_3]
        min_distortion_confidence_top_3 = min_distortion_confidence[min_distortion_top_3]

        # Instead of the existing print statements for confidence rates, add:
        min_epsilon_top_3_data = list(zip(min_epsilon_top_3, min_epsilon_confidence_top_3))
        min_top_n_top_3_data = list(zip(min_top_n_top_3, min_top_n_confidence_top_3))
        min_distortion_top_3_data = list(zip(min_distortion_top_3, min_distortion_confidence_top_3))

        # Display confidence text in a separate window
        #plot_confidence_text(min_epsilon_top_3_data, min_top_n_top_3_data, min_distortion_top_3_data)

        # Create a new figure with a specific figure number
        # plt.clf()  
        # fig = plt.figure(num=2, figsize=(10, 7))
        fig = Figure(figsize=(9,6.5))#, num=2)
        #axes = fig.subplots(1,2)
        # fig.suptitle("Minimum Successful Adversarial Examples", y=0.95)

        # Case 1: Minimum epsilon successful case
        axes = fig.subplots(3, 4)

        axes[0, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')

        axes[0, 1].imshow(min_epsilon_adv_image_np.reshape(28, 28), cmap='gray')
        #axes[0, 1].set_title(f"Adversarial Image\nMin Epsilon: {round(min_epsilon, 1)}, Top N: {min_epsilon_top_n}")
        axes[0, 1].set_title(f"Min Epsilon Case ({round(min_epsilon, 1)})\nAdversarial Image")
        axes[0, 1].axis('off')

        axes[0, 2].imshow(perturbation_min_epsilon.reshape(28, 28), cmap='hot')
        #axes[0, 2].set_title("Perturbation (Min Epsilon)")
        axes[0, 2].set_title("Perturbation")
        axes[0, 2].axis('off')

        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        axes[0, 3].set_title("Confidence Scores (%)")
        bar_colors = ['blue'] * 10
        bar_colors[original_label[0]] = 'green'
        bar_colors[target_label[0]] = 'red'
        min_epsilon_confidence = [v * 100 for v in min_epsilon_confidence]
        axes[0, 3].bar(digits, min_epsilon_confidence, color=bar_colors)

        # Case 2: Minimum pixel count successful case
        axes[1, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
        axes[1, 0].set_title("Original Image")
        axes[1, 0].axis('off')

        axes[1, 1].imshow(min_top_n_adv_image_np.reshape(28, 28), cmap='gray')
        #axes[1, 1].set_title(f"Adversarial Image\nMin Top N: {min_top_n}, Epsilon: {round(min_top_n_epsilon, 1)}")
        axes[1, 1].set_title(f"Min Pixels Case ({min_top_n})\nAdversarial Image")
        axes[1, 1].axis('off')

        axes[1, 2].imshow(perturbation_min_top_n.reshape(28, 28), cmap='hot')
        # axes[1, 2].set_title("Perturbation (Min Top N)")
        axes[1, 2].set_title("Perturbation")
        axes[1, 2].axis('off')

        axes[1, 3].set_title("Confidence Scores (%)")
        bar_colors = ['blue'] * 10
        bar_colors[original_label[0]] = 'green'
        bar_colors[target_label[0]] = 'red'
        min_top_n_confidence = [v * 100 for v in min_top_n_confidence]
        axes[1, 3].bar(digits, min_top_n_confidence, color=bar_colors)

        # Case 3: Minimum distortion successful case
        axes[2, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
        axes[2, 0].set_title("Original Image")
        axes[2, 0].axis('off')

        axes[2, 1].imshow(min_distortion_adv_image_np.reshape(28, 28), cmap='gray')
        #axes[2, 1].set_title(f"Adversarial Image\nMin Distortion: {round(float(min_distortion.numpy()*100), 2)}%, Top N: {round(min_distortion_top_n,1)}, Epsilon: {round(min_distortion_epsilon, 1)}")
        axes[2, 1].set_title(f"Min Distortion Case ({round(float(min_distortion.numpy()*100), 2)}%)\nAdversarial Image")
        axes[2, 1].axis('off')

        axes[2, 2].imshow(perturbation_min_distortion.reshape(28, 28), cmap='hot')
        # axes[2, 2].set_title("Perturbation (Min Top N)")
        axes[2, 2].set_title("Perturbation")
        axes[2, 2].axis('off')

        axes[2, 3].set_title("Confidence Scores (%)")
        bar_colors = ['blue'] * 10
        bar_colors[original_label[0]] = 'green'
        bar_colors[target_label[0]] = 'red'
        min_distortion_confidence = [v * 100 for v in min_distortion_confidence]
        axes[2, 3].bar(digits, min_distortion_confidence, color=bar_colors)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5)

        #display_frame = Frame(root, background="white", relief="sunken")

        canvas = FigureCanvasTkAgg(fig, master=display_frame)
        canvas.draw()

        legend_frame = Frame(display_frame, bd=2, background="white")#, relief="sunken")
        legend_label = Label(legend_frame, text="Green: Source Class         Red: Target Class         Blue: Other Classes", bg="white", fg="black", font=("Arial", 25, "bold"))

        canvas.get_tk_widget().pack(side="top", fill="both")
        legend_label.pack(pady=2)
        legend_frame.pack(side="bottom",fill="both", expand=True)
    
    #display_frame.update_idletasks()

    display_frame.pack(fill="both", expand=True)

    

    

    



def clear_screen(root):
   for widget in root.winfo_children():
        widget.destroy()


def main():
    valid_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def specific_test(model, sample_image, sample_label, target_label, epsilon, root):
        print(f"\nGenerating adversarial example for digit: {sample_label} using Saliency map")
        start = time.perf_counter()
        adversarial_image, num_pixels_changed= create_adversarial_example_saliency(root, model, sample_image, sample_label,
                                                                target_label, epsilon, 0, convolutional)
        end = time.perf_counter()

        print(f"Adversarial example generation took {(end - start):.3f} seconds.")

        # Get predictions
        original_pred, original_probs = predict_sample(model, sample_image)
        adv_pred, adv_probs = predict_sample(model, adversarial_image)

        # Print predictions
        print(f"\nOriginal Prediction: {original_pred}")
        print(f"Top 3 probabilities for original image:")
        top3_orig = np.argsort(original_probs)[-3:][::-1]
        for digit, prob in zip(top3_orig, original_probs[top3_orig]):
            print(f"Digit {digit}: {prob * 100:.2f}%")

        print(f"\nAdversarial Prediction: {adv_pred}")
        print(f"Top 3 probabilities for adversarial image:")
        top3_adv = np.argsort(adv_probs)[-3:][::-1]
        for digit, prob in zip(top3_adv, adv_probs[top3_adv]):
            print(f"Digit {digit}: {prob * 100:.2f}%")

        # Add success/failure message
        if adv_pred == target_label:
            print(f"\nSuccess! Model was fooled: {sample_label} → {target_label}")
            distortion = tf.reduce_sum(abs(sample_image - adversarial_image)) / float(784)
            print("Input feature distortion: ", round(float(distortion.numpy()*100), 2), "%")
        else:
            print("\nThe model wasn't fooled. Try increasing epsilon.")

        # Visualize the results
        difference = adversarial_image - sample_image
        #_, adv_probs = predict_sample(model, adversarial_image)
        plot_images(root, sample_image, sample_label, adversarial_image, target_label, difference, adv_pred, adv_probs)



    def iteration_test(model, processed_image, target_label, root):

        # Get prediction for original image
        original_pred, original_probs = predict_sample(model, processed_image)
        print("\nOriginal Prediction:")
        print(f"Predicted digit: {original_pred}")
        print("\nTop 3 probabilities:")
        top3 = np.argsort(original_probs)[-3:][::-1]
        for digit, prob in zip(top3, original_probs[top3]):
            print(f"Digit {digit}: {prob * 100:.2f}%")

        # Iterate over values
        success_rates = []
        for epsilon in np.arange(0.1, 1.1, 0.1):
            #print(f"Testing epsilon={epsilon:.1f}, Top N={top_n}")
            print(f"Testing epsilon={epsilon:.1f}")
            #adversarial_image, num_pixels_changed = create_adversarial_example_saliency(tab1, model, processed_image, original_pred,
            #                                                        target_label, epsilon, 150, convolutional)
            adversarial_image, num_pixels_changed = create_adversarial_example_saliency(root, model, processed_image, original_pred,
                                                                    target_label, epsilon, 150, convolutional)
            #print(num_pixels_changed)
            # Get prediction for adversarial image
            original_pred, original_probs = predict_sample(model, processed_image)
            adv_pred, adv_probs = predict_sample(model, adversarial_image)

            success = 1 if adv_pred == target_label else 0
            distortion = 0
            if success:
                print("Adversarial example can be made with epsilon ", epsilon, ", ", num_pixels_changed, " had to be perturbed.")
                distortion = tf.reduce_sum(abs(processed_image - adversarial_image)) / float(784)
                print(distortion)
            else:
                print("Adversarial example can NOT be made with epsilon", epsilon)
            for n in range(0, 201, 5):
                if n < num_pixels_changed:
                    success_rates.append((epsilon, n, distortion, 0, adv_probs, adversarial_image))
                else:
                    success_rates.append((epsilon, n, distortion, success, adv_probs, adversarial_image))

        clear_screen(root)
        # menu button
        # menu_btn_frame = Frame(root, pady=5, bd=2, background="gray")
        # menu_btn = Button(menu_btn_frame, text="Reset!", command=lambda: load_menu(), bg="white", fg="black")
        # menu_btn.pack(expand=True, fill="y")
        # menu_btn_frame.pack()
        reset_btn = Button(root, text="Reset!", command=lambda: load_menu(),
                  bg="white", fg="black", bd=2, highlightthickness=0, relief="solid", font=("Arial", 30, "bold"))
        reset_btn.pack(pady=5)#, expand=True, fill="y")
        
        # style = ttk.Style()
        # style.theme_use('default')

        # # # Use the default "gray" color
        # style.configure('TNotebook', background='gray')              # Tab bar background
        # style.configure('TNotebook.Tab', background='gray')          # Inactive tabs
        # style.map('TNotebook.Tab', background=[('selected', 'light gray')])  # Selected tab

        tabControl = ttk.Notebook(root)

        #tab1 = ttk.Frame(tabControl)
        tab2 = Frame(tabControl, background="gray")
        tab3 = Frame(tabControl, background="gray")
        # tab2 = Frame(tabControl, bg="gray", bd=2, relief="sunken")
        # tab3 = Frame(tabControl, bg="gray", bd=2, relief="sunken")

        #tabControl.add(tab1, text='Adversarial Image')
        tabControl.add(tab2, text='Success Rates')
        tabControl.add(tab3, text='Best Cases')

        tabControl.pack(expand=True, fill='both')

        #plot_images(root, processed_image, original_pred, adversarial_image, target_label, difference, adv_probs)

        # Plot success rate heatmap first, with block=False
        plot_success_rate(success_rates, tab2)

        # Plot minimum success cases second, with block=True
        plot_min_success_cases(success_rates, model, processed_image, np.array([original_pred]),
                                [target_label], tab3)

        # Keep both windows open until any key is pressed
        #plt.show()
    
    def plot_images(root, original, input_label, adversarial, target_label, difference, adv_pred, adv_probs):
        """Plot the original image, adversarial image, and their difference."""
        # Convert to numpy if tensor
        if tf.is_tensor(original):
            original = original.numpy()
        if tf.is_tensor(adversarial):
            adversarial = adversarial.numpy()
        if tf.is_tensor(difference):
            difference = difference.numpy()

        # Reshape images for display
        original = original.reshape(28, 28)
        adversarial = adversarial.reshape(28, 28)
        difference = difference.reshape(28, 28)

        clear_screen(root)
        # menu button
        # menu_btn = ttk.Button(root, text="Reset!", command=lambda: load_menu())
        # menu_btn.pack()
        reset_btn = Button(root, text="Reset!", command=lambda: load_menu(),
                  bg="white", fg="black", bd=2, highlightthickness=0, relief="solid", font=("Arial", 30, "bold"))
        reset_btn.pack(pady=5)#, expand=True, fill="y")

        distortion = tf.reduce_sum(abs(original - adversarial)) / float(784)

        success_text = f"Attack success!\nInput feature distortion: {round(float(distortion.numpy()*100), 2)}%"
        if (adv_pred != target_label):
            success_text = "Attack was not successful."
        success_frame = Frame(root, bd=2, background="white", relief="sunken")
        success_label = Label(success_frame, text=success_text, bg="white", fg="black", font=("Arial", 25, "bold"))
        success_label.pack(pady=2)
        success_frame.pack(expand=True, fill="both")

        plt.clf()  
        legend_frame = Frame(success_frame, background="white")#, relief="sunken")
        legend_label = Label(legend_frame, text="Green: Source Class         Red: Target Class         Blue: Other Classes", bg="white", fg="black", font=("Arial", 25, "bold"))
        legend_label.pack()
        legend_frame.pack(expand=True, side="bottom",fill="both")
        fig = Figure(figsize=(5,6.75))
        #fig.patch.set_facecolor("none")
        axes = fig.subplots(2,2)
        canvas = FigureCanvasTkAgg(fig, master = success_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="x")

        #fig, axes = plt.subplots(1, 4, figsize=(15, 5))

        # Plot original image
        im1 = axes[0,0].imshow(original, cmap='gray')
        axes[0,0].set_title("Original Image")
        axes[0,0].axis('off')
        fig.colorbar(im1, ax=axes[0,0])

        # Plot adversarial image
        im2 = axes[0,1].imshow(adversarial, cmap='gray')
        axes[0,1].set_title("Adversarial Image")
        axes[0,1].axis('off')
        fig.colorbar(im2, ax=axes[0,1])

        # Plot the difference (noise)
        # Normalize difference to symmetric range for seismic colormap
        max_diff = max(abs(difference.max()), abs(difference.min()))
        im3 = axes[1,0].imshow(difference, cmap='seismic', vmin=-max_diff, vmax=max_diff)
        axes[1,0].set_title("Perturbation (Difference)")
        axes[1,0].axis('off')
        fig.colorbar(im3, ax=axes[1,0])

        axes[1,1].set_title("Confidence Scores (%)")
        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        bar_colors = ['blue'] * 10
        bar_colors[input_label] = 'green'
        bar_colors[target_label] = 'red'
        adv_probs = [v * 100 for v in adv_probs]
        axes[1,1].bar(digits, adv_probs, color=bar_colors)

        fig.tight_layout()
        # plt.tight_layout()
        # plt.show()
    
    def get_mnist_input(input_panel, source_choice_window, x_test, y_test, source_digit):
        """
        Get a sample image of the specified digit from the test set.

        Parameters:
        x_test (numpy.ndarray): The test images dataset
        y_test (numpy.ndarray): The test labels dataset
        source_digit (int): The digit to find (0-9)

        Returns:
        tuple: (sample_image, sample_label) where sample_image is the first instance
            of the target digit found in the test set
        """
        # Find all indices where the label matches the target digit
        digit_indices = np.where(y_test == source_digit)[0]

        if len(digit_indices) == 0:
            raise ValueError(f"No instances of digit {source_digit} found")

        # Select random index instead of first one
        sample_idx = np.random.choice(digit_indices)
        #return x_test[sample_idx:sample_idx + 1], y_test[sample_idx:sample_idx + 1]
        nonlocal input_image
        input_image = x_test[sample_idx:sample_idx + 1]
        display_input_image(input_panel)
        # fig = Figure(figsize=(1,1))
        # axes = fig.subplots(1,1)

        # # Creating the Tkinter canvas containing the Matplotlib figure
        # clear_screen(input_panel)
        # canvas = FigureCanvasTkAgg(fig, master = input_panel)
        # canvas.draw()
        # canvas.get_tk_widget().pack(expand=True, fill='both')

        # # setting title
        # #plt.title("Live Adversarial Attack", fontsize=20)
        # #fig.set_title(f"Input Image")
        # #fig.axis('off')
        # axes.set_title(f"Input Image")
        # axes.axis('off')
        # im = axes.imshow(input_image.reshape(28, 28), cmap='gray')

        # source_choice_window.destroy()


    
    
    # Initialize GUI
    root = tk.Tk()
    root.title("Adversarial Attacks")
    width= root.winfo_screenwidth()               
    height= root.winfo_screenheight()               
    root.geometry("%dx%d" % (width, height))

    # input image
    input_image = None


    # Welcome screen
    # clear_screen(root)
    # welcome_screen_frame = ttk.Frame(root, padding=10)
    # welcome_screen_frame.pack(expand=True, fill="both")
    # label_1 = tk.Label(master = welcome_screen_frame, text="Welcome to the Adversarial Attack Program!", font=("Arial", 14), justify="center")
    # label_1.pack()
    # label_2 = tk.Label(master = welcome_screen_frame, text="Model training.....", font=("Arial", 14), justify="center")
    # label_2.pack()
    
    # Load and preprocess MNIST dataset for training
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print("Done!")

    # Normalize and reshape data
    x_train, x_test = (x_train/255.0).astype(np.float32), (x_test/255.0).astype(np.float32)

    # Model and parameters
    model = None
    convolutional = None
    train_model = None
    baseline_accuracy = 0
    
    def initialize_model():
        # Parameters
        # convolutional = False    # True: CNN, False: MLP
        # train_model = False       # True: training model, False: loading model
        nonlocal x_train, x_test, convolutional, train_model, model, baseline_accuracy

        if convolutional:
            x_train = x_train.reshape((-1,28,28,1))
            x_test = x_test.reshape((-1,28,28,1))
        else:
            x_train = x_train.reshape(-1, 784)
            x_test = x_test.reshape(-1, 784)
        
        if train_model:
            # Train the model
            print("Training the model...")
            if convolutional:
                model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
                model.save('cnn_model.keras')
            else:
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(784,)),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(10, activation='softmax')
                ])
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)
                model.save('mlp_model.keras')
        else:
            # Load the model
            print("Loading the model...")
            if convolutional:
                model = tf.keras.models.load_model('cnn_model.keras')
            else:
                model = tf.keras.models.load_model('mlp_model.keras')

        baseline_accuracy = model.evaluate(x_test, y_test)[1]
        print("Model baseline accuracy: ", baseline_accuracy) 
        # Show model architecture
        # model.summary()

    
    # def test_mnist_iterate():
    #     def mnist_iterate(source_digit, target_label):
    #         try:
    #             # Get a sample image of the requested digit
    #             sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)
    #             iteration_test(model, sample_image, target_label, root)
    #         except ValueError as e:
    #             print(f"Error: {e}")

    #     # new_root.mainloop()
    #     clear_screen(root)
    
    #     # generate new frame
    #     test_mnist_iterate_frame = ttk.Frame(root, padding=10)
    #     test_mnist_iterate_frame.pack(expand=True, fill="x")

    #     # instruction label 1
    #     instruction_label_1 = tk.Label(master = test_mnist_iterate_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
    #     instruction_label_1.pack()

    #     # source class entry
    #     source_digit = tk.StringVar()
    #     source_digit.set(valid_digits[0])

    #     # target class entry
    #     target_class = tk.StringVar()
    #     target_class.set(valid_digits[1])

    #     source_label = ttk.Label(test_mnist_iterate_frame, text="Source Class:")
    #     source_label.pack()
    #     source_dropdown = ttk.OptionMenu(test_mnist_iterate_frame, source_digit, *valid_digits)
    #     source_dropdown.pack()

    #     target_label = ttk.Label(test_mnist_iterate_frame, text="Target Class:")
    #     target_label.pack()
    #     target_dropdown = ttk.OptionMenu(test_mnist_iterate_frame, target_class, *valid_digits)
    #     target_dropdown.pack()

    #     # Attack button
    #     button = ttk.Button(master=test_mnist_iterate_frame, text="Run Attack!",
    #                             command=lambda: mnist_iterate(int(source_digit.get()),
    #                                                         int(target_class.get())))
    #     button.pack(pady=5)

    #     # menu button
    #     menu_btn = ttk.Button(test_mnist_iterate_frame, text="Reset!", command=lambda: load_menu())
    #     menu_btn.pack(pady = 50)


    # def test_uploaded_iterate():
    #     def file_option(target_label):
    #         # get image from file system
    #         file_path = filedialog.askopenfilename()
    #         try:
    #             processed_image = preprocess_uploaded_image(file_path, convolutional)
    #             iteration_test(model, processed_image, target_label, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")   

    #     def draw_option(target_label):
    #         # get image from drawing
    #         try:
    #             processed_image = get_drawn_digit(convolutional)
    #             iteration_test(model, processed_image, target_label, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")

    #     def picture_option(target_label):
    #         # Test with uploaded image or drawing
    #         try:
    #             processed_image = preprocess_uploaded_image('/home/designteam10/Pictures/image.jpg', convolutional)
    #             iteration_test(model, processed_image, target_label, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")

    #     '''def base_picture_option(target_label):
    #         # Test with uploaded images
    #         try:
    #             processed_image = upload_image_with_base('/home/designteam10/Pictures/image.jpg')
    #             iteration_test(model, processed_image, target_label, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")'''


    #     clear_screen(root)

    #     # generate new frame
    #     test_uploaded_iterate_frame = ttk.Frame(root, padding=10)
    #     test_uploaded_iterate_frame.pack(expand=True, fill="x")

    #     # instruction label 1
    #     instruction_label_1 = tk.Label(master = test_uploaded_iterate_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
    #     instruction_label_1.pack()

    #     # target class entry
    #     target_class = tk.StringVar()
    #     target_class.set(valid_digits[1])

    #     input_frame1 = ttk.Frame(master=test_uploaded_iterate_frame)
    #     target_label = ttk.Label(test_uploaded_iterate_frame, text="Target Class:")
    #     target_label.pack()
    #     target_dropdown = ttk.OptionMenu(test_uploaded_iterate_frame, target_class, *valid_digits)
    #     target_dropdown.pack()
    #     input_frame1.pack(pady=5)

    #     # Instruction label 2
    #     instruction_label_2 = tk.Label(master = test_uploaded_iterate_frame, text="Choose one of the options below to run the attack:", font=("Arial", 14), justify="center")
    #     instruction_label_2.pack()

    #     # Create upload button
    #     upload_btn = ttk.Button(test_uploaded_iterate_frame, text="Upload Image", command=lambda: file_option(int(target_class.get())))
    #     upload_btn.pack(pady=5)

    #     # Draw digit button
    #     draw_btn = ttk.Button(test_uploaded_iterate_frame, text="Draw Digit", command=lambda: draw_option(int(target_class.get())))
    #     draw_btn.pack(pady=5)

    #     # Take picture button
    #     picture_btn = ttk.Button(test_uploaded_iterate_frame, text="Take Picture", command=lambda: picture_option(int(target_class.get())))
    #     picture_btn.pack(pady=5)

    #     # menu button
    #     menu_btn = ttk.Button(test_uploaded_iterate_frame, text="Reset!", command=lambda: load_menu())
    #     menu_btn.pack(pady = 50)



    # def test_mnist_specify():
    #     def mnist_specify(source_digit, target_label, epsilon):
    #         try:
    #             # Get a sample image of the requested digit
    #             sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)
    #             specific_test(model, sample_image, sample_label[0], target_label, epsilon, root)
    #         except ValueError as e:
    #             print(f"Error: {e}")

    #     clear_screen(root)
    
    #     # generate new frame
    #     test_mnist_specify_frame = ttk.Frame(root, padding=10)
    #     test_mnist_specify_frame.pack(expand=True, fill="x")

    #     # instruction label 1
    #     instruction_label_1 = tk.Label(master = test_mnist_specify_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
    #     instruction_label_1.pack()

    #     # source class entry
    #     source_digit = tk.StringVar()
    #     source_digit.set(valid_digits[0])

    #     # target class entry
    #     target_class = tk.StringVar()
    #     target_class.set(valid_digits[1])

    #     source_label = ttk.Label(test_mnist_specify_frame, text="Source Class:")
    #     source_label.pack()
    #     source_dropdown = ttk.OptionMenu(test_mnist_specify_frame, source_digit, *valid_digits)
    #     source_dropdown.pack()

    #     target_label = ttk.Label(test_mnist_specify_frame, text="Target Class:")
    #     target_label.pack()
    #     target_dropdown = ttk.OptionMenu(test_mnist_specify_frame, target_class, *valid_digits)
    #     target_dropdown.pack()


    #     # epsilon entry
    #     input_frame3 = ttk.Frame(master=test_mnist_specify_frame)
    #     epsilon_label = ttk.Label(input_frame3, text="Epsilon:")
    #     epsilon_label.pack()
    #     epsilon_slider = tk.Scale(input_frame3, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200)
    #     epsilon_slider.pack()
    #     input_frame3.pack(pady=5)

    #     # Attack button
    #     button = ttk.Button(master=test_mnist_specify_frame, text="Run Attack!",
    #                             command=lambda: mnist_specify(int(source_digit.get()),
    #                                                         int(target_class.get()),
    #                                                         float(epsilon_slider.get())))
    #     button.pack(pady=10)

    #     # menu button
    #     menu_btn = ttk.Button(test_mnist_specify_frame, text="Reset!", command=lambda: load_menu())
    #     menu_btn.pack(pady = 50)


    # def test_uploaded_specify():
    #     def file_option(target_label, epsilon):
    #         file_path = filedialog.askopenfilename()
    #         try:
    #             processed_image = preprocess_uploaded_image(file_path, convolutional)
    #             pred, probs = predict_sample(model, processed_image)
    #             specific_test(model, processed_image, pred, target_label, epsilon, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")

    #     def draw_option(target_label, epsilon):
    #         try:
    #             processed_image = get_drawn_digit(convolutional)
    #             pred, probs = predict_sample(model, processed_image)
    #             specific_test(model, processed_image, pred, target_label, epsilon, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")

    #     def picture_option(target_label, epsilon):
    #         try:
    #             processed_image = preprocess_uploaded_image('/home/designteam10/Pictures/image.jpg', convolutional)
    #             #print(processed_image)
    #             pred, probs = predict_sample(model, processed_image)
    #             specific_test(model, processed_image, pred, target_label, epsilon, root)
    #         except Exception as e:
    #             print(f"Error processing image: {str(e)}")

    #     clear_screen(root)
    
    #     # generate new frame
    #     test_uploaded_specify_frame = ttk.Frame(root, padding=10)
    #     test_uploaded_specify_frame.pack(expand=True, fill="x")

    #     # instruction label 1
    #     instruction_label_1 = tk.Label(master = test_uploaded_specify_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
    #     instruction_label_1.pack()

    #     # epsilon entry
    #     input_frame1 = ttk.Frame(master=test_uploaded_specify_frame)
    #     epsilon_label = ttk.Label(input_frame1, text="Epsilon:")
    #     epsilon_label.pack()
    #     epsilon_slider = tk.Scale(input_frame1, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200)
    #     epsilon_slider.pack(padx=10)
    #     input_frame1.pack(pady=5)

    #     # target class entry
    #     target_class = tk.StringVar()
    #     target_class.set(valid_digits[1])

    #     input_frame3 = ttk.Frame(master=test_uploaded_specify_frame)
    #     target_label = ttk.Label(input_frame3, text="Target Class:")
    #     target_label.pack()
    #     target_dropdown = ttk.OptionMenu(input_frame3, target_class, *valid_digits)
    #     target_dropdown.pack()
    #     input_frame3.pack(pady=5)

    #     # Instruction label 2
    #     instruction_label_2 = tk.Label(master = test_uploaded_specify_frame, text="Choose one of the options below to run the attack:", font=("Arial", 14), justify="center")
    #     instruction_label_2.pack()

    #     # Create upload button
    #     upload_btn = ttk.Button(test_uploaded_specify_frame, text="Upload Image", command=lambda: file_option(int(target_class.get()),
    #                                                                                         float(epsilon_slider.get())))
    #     upload_btn.pack(pady=5)

    #     # Draw digit button
    #     draw_btn = ttk.Button(test_uploaded_specify_frame, text="Draw Digit", command=lambda: draw_option(int(target_class.get()),
    #                                                                                     float(epsilon_slider.get())))
    #     draw_btn.pack(pady=5)

    #     # Take picture button
    #     picture_btn = ttk.Button(test_uploaded_specify_frame, text="Take Picture", command=lambda: picture_option(int(target_class.get()),
    #                                                                                     float(epsilon_slider.get())))
    #     picture_btn.pack(pady=5)

    #     # menu button
    #     #menu_frame = ttk.Frame(master=test_uploaded_specify_frame)
    #     menu_btn = ttk.Button(test_uploaded_specify_frame, text="Reset!", command=lambda: load_menu())
    #     menu_btn.pack(pady = 50)
    #     # menu_frame.pack(expand=True,fill="x")

    # Option screen
    def load_menu():
        # clear_screen(root)
        # option_screen_frame = ttk.Frame(root, padding=10)
        # option_screen_frame.pack(expand=True, fill="x")
        # option_label = tk.Label(master = option_screen_frame, text="Choose an option:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        # option_label.pack()
        # button1 = ttk.Button(master = option_screen_frame, text="Test MNIST digits across multiple epsilons", command=test_mnist_iterate)#.grid(column=0, row=1)
        # button1.pack(padx=20, pady=5)
        # button2 = ttk.Button(master = option_screen_frame, text="Test uploaded image across multiple epsilons", command=test_uploaded_iterate)#.grid(column=0, row=2)
        # button2.pack(padx=20, pady=5)
        # button3 = ttk.Button(master = option_screen_frame, text="Create an adversarial example for random number from MNIST dataset (specify max epsilon)", command=test_mnist_specify)#.grid(column=0, row=3)
        # button3.pack(padx=20, pady=5)
        # button4 = ttk.Button(master = option_screen_frame, text="Create an adversarial example for uploaded picture (specify max epsilon)", command=test_uploaded_specify)#.grid(column=0, row=4)
        # button4.pack(padx=20, pady=5)
        # tk.Label(master = option_screen_frame, text=" ", font=("Arial", 20), justify="center").pack()
        # button5 = ttk.Button(master = option_screen_frame, text="Choose new neural network model", command=lambda: load_welcome_screen())
        # button5.pack()
        # tk.Label(master = option_screen_frame, text=" ", font=("Arial", 6), justify="center").pack()
        # button6 = ttk.Button(master = option_screen_frame, text="Exit program", command=root.destroy)
        # button6.pack()

        #plt.close('all')

        target_class = tk.StringVar()
        target_class.set(valid_digits[0])

        nonlocal input_image
        nonlocal convolutional
        nonlocal baseline_accuracy
        input_image = None

        def run_iterate_attack(target):
            nonlocal input_image
            nonlocal attack_panel
            # clear_screen(attack_panel)
            # attack_header_frame = Frame(attack_panel, pady=5, bd=2, background="gray")
            # attack_header_label = Label(attack_header_frame, text="Live Adversarial Attack Visualization", bg ="gray", fg="black", font=("Arial", 25, "bold"))
            # attack_header_label.pack()
            # attack_header_frame.pack()
            pred, probs = predict_sample(model, input_image)
            iteration_test(model, input_image, target, attack_panel)

        def run_specific_attack(target, epsilon):
            nonlocal input_image
            nonlocal attack_panel
            # clear_screen(attack_panel)
            # attack_header_frame = Frame(attack_panel, pady=5, bd=2, background="gray")
            # attack_header_label = Label(attack_header_frame, text="Live Adversarial Attack Visualization", bg ="gray", fg="black", font=("Arial", 25, "bold"))
            # attack_header_label.pack()
            # attack_header_frame.pack()
            pred, probs = predict_sample(model, input_image)
            specific_test(model, input_image, pred, target, epsilon, attack_panel)

        def load_iterate_parameters():
            clear_screen(parameter_select_frame)
            iterate_frame = Frame(parameter_select_frame, pady=5, bd=2, background="gray")
            target_label = Label(iterate_frame, text="Target Class:", bg ="gray", fg="black", font=("Arial", 25, "bold"))
            target_label.pack()
            target_dropdown = OptionMenu(iterate_frame, target_class, *valid_digits)
            target_dropdown.config(bg ="gray", fg="black", font=("Arial", 20, "bold"))
            target_dropdown.pack()
            iterate_frame.pack(pady=5, expand=True, fill="both")
            
            iterate_button_frame = Frame(iterate_frame, bd=2, background="gray")
            iterate_button_frame.pack(expand=True, fill="both")

            run_attack_btn = Button(iterate_frame, text="Run Attack!", command=lambda: run_iterate_attack(int(target_class.get())),
                                             bg="gray", fg="black", font=("Arial", 30, "bold"))
            run_attack_btn.pack(padx=20, pady=20, expand=True)#, fill="y")
            root.update_idletasks()
        
        def load_specific_parameters():
            clear_screen(parameter_select_frame)
            specific_frame = Frame(parameter_select_frame, pady=5, bd=2, background="gray")
            target_label = Label(specific_frame, text="Target Class:", bg="gray", fg="black", font=("Arial", 25, "bold"))
            target_label.pack()
            target_dropdown = OptionMenu(specific_frame, target_class, *valid_digits)
            target_dropdown.config(bg ="gray", fg="black", font=("Arial", 20, "bold"))
            target_dropdown.pack()
            epsilon_label = Label(specific_frame, text="Epsilon:", bg="gray", fg="black", font=("Arial", 25, "bold"))
            epsilon_label.pack()
            epsilon_slider = tk.Scale(specific_frame, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200, bg ="gray", fg="black", font=("Arial", 15, "bold"))
            epsilon_slider.pack(padx=10)
            specific_frame.pack( expand=True, fill="both")

            specific_button_frame = Frame(specific_frame, pady=5, bd=2, background="gray")
            specific_button_frame.pack(expand=True, fill="both")

            run_attack_btn = Button(specific_button_frame, text="Run Attack!", command=lambda: run_specific_attack(int(target_class.get()),
                                                            float(epsilon_slider.get())), bg="gray", fg="black", font=("Arial", 30, "bold"))
            run_attack_btn.pack(padx=20, pady=10, expand=True)#, fill="y")
            root.update_idletasks()

        def on_radio_change():
            if iterate_var.get():
                load_iterate_parameters()
            else:
                load_specific_parameters()


        clear_screen(root)

        # Configure grid layout for the root window
        root.rowconfigure(0, weight=2)  # Make top-left panel smalle
        root.rowconfigure(1, weight=5, minsize=200)  # Make top-left panel smaller
        root.rowconfigure(2, weight=16)
        #root.rowconfigure(3, weight=2)
        root.columnconfigure(0, weight=1, minsize=150)  # Make top-left panel smaller
        root.columnconfigure(1, weight=2)
        
        # Function to create a panel with an inner box
        def create_panel(parent, row, column, rowspan=1, columnspan=1):
            outer_frame = Frame(parent, bg="gray", bd=2, relief="sunken")
            outer_frame.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky="nsew", padx=5,
                             pady=5)

            inner_frame = Frame(outer_frame, bg="gray", bd=2, relief="ridge")
            inner_frame.pack(fill="both", expand=True, padx=10, pady=10)

            return inner_frame
        
        def quit_program(root):
            root.destroy()
            exit()

        digit_options = ['MNIST Digit', 'Draw Digit', 'Upload Image', 'Camera Capture']

        # Create panel with options
        option_panel = create_panel(root, row=0, column=0)
        option_panel.grid_columnconfigure(0, weight=1)
        option_panel.grid_columnconfigure(1, weight=1)
        option_panel.grid_rowconfigure(0, weight=1)
        choose_model_button = tk.Button(option_panel, text='← Go Back to Welcome Screen', font=("Arial", 15, "bold"), command=lambda:load_welcome_screen())
        choose_model_button.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        quit_button = tk.Button(option_panel, text='Quit Program', font=("Arial", 15, "bold"), command=lambda:quit_program(root))
        quit_button.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        option_panel.grid_propagate(False)
        # model_instruction_label = tk.Label(master = input_panel, text="Choose input format:", font=("Arial", 20, "bold")
        #                                 , justify="center", background="gray", fg="black", underline=0)
        # input_instruction_label.grid(row=0, column=0, pady=2, sticky="nsew")

        # Create input panel
        input_panel = create_panel(root, row=1, column=0)

        # Configure the input_panel for centering
        input_panel.grid_columnconfigure(0, weight=1)  # Center align items horizontally
        input_panel.grid_rowconfigure(tuple(range(len(digit_options)+1)), weight=1)  # Distribute space equally

        # underline_font = tk.font.Font(family="Arial", size=25, weight="bold", underline=1)

        input_instruction_label = tk.Label(master = input_panel, text="Choose input format:", font=("Arial", 20, "bold")
                                        , justify="center", background="gray", fg="black", underline=0)
        input_instruction_label.grid(row=0, column=0, pady=2, sticky="nsew")

        # Create buttons in the top-left panel and center them
        mnist_btn = tk.Button(input_panel, text='Random Sample from MNIST Dataset', font=("Arial", 20, "bold"), command=lambda:get_input_image(input_panel, "mnist"))
        mnist_btn.grid(row=1, column=0, pady=2, padx=2, sticky="nsew")

        draw_btn = tk.Button(input_panel, text='Draw Digit', font=("Arial", 20, "bold"), command=lambda:get_input_image(input_panel, "draw"))
        draw_btn.grid(row=2, column=0, pady=2, padx=2, sticky="nsew")

        upload_btn = tk.Button(input_panel, text='Upload Image', font=("Arial", 20, "bold"), command=lambda:get_input_image(input_panel, "upload"))
        upload_btn.grid(row=3, column=0, pady=2, padx=2, sticky="nsew")

        camera_btn = tk.Button(input_panel, text='Camera Capture', font=("Arial", 20, "bold"), command=lambda:get_input_image(input_panel, "camera"))
        camera_btn.grid(row=4, column=0, pady=2, padx=2, sticky="nsew")

        input_panel.grid_propagate(False)

        # Create bottom-left panel
        parameter_panel = create_panel(root, row=2, column=0)
        parameter_instruction_label = tk.Label(master = parameter_panel, text="Set Attack Parameters:", font=("Arial", 20, "bold"),
                                               justify="center", background="gray", fg="black")

        parameter_instruction_label.pack(pady=7)
        iterate_option_frame = Frame(parameter_panel, pady=5, relief="sunken", bd=2, background="gray")
        iterate_option_frame.pack(fill="x", side="top")

        instruction_label = tk.Label(master = iterate_option_frame, text="Choose type of adversarial test:", font=("Arial", 25, "bold"),
                                     justify="center", background="gray", fg="black")#.grid(column=0, row=0)
        instruction_label.pack(pady=5)
        iterate_var = tk.BooleanVar(value = False)
        radio_specific = Radiobutton(iterate_option_frame, text="Test specific epsilon", variable=iterate_var, value=False, bg="gray", fg="black", font=("Arial", 20, "bold"), command=on_radio_change)
        radio_specific.pack()
        radio_iterate = Radiobutton(iterate_option_frame, text="Iterate through epsilons", variable=iterate_var, value=True, bg="gray", fg="black", font=("Arial", 20, "bold"), command=on_radio_change)
        radio_iterate.pack()
        parameter_select_frame = Frame(parameter_panel, relief="sunken", bd=2, background="gray")
        parameter_select_frame.pack(expand=True,fill="both")
        parameter_panel.propagate(False)
        load_specific_parameters()

        # Create right panel
        attack_panel = create_panel(root, row=0, column=1, rowspan=3)
        if convolutional:
            matrix_image = Image.open("Images/cnn_matrix.png")
            model_name = "CNN"
        else:
            matrix_image = Image.open("Images/mlp_matrix.png")
            model_name = "MLP"
        matrix_image = matrix_image.resize((660,528))
        photo = ImageTk.PhotoImage(matrix_image)
        attack_caption_label = tk.Label(master = attack_panel, text=f"{model_name} model baseline accuracy: {round(float(baseline_accuracy*100), 2)}%\n\nThis heatmap shows the success rate (%) of the\nadversarial attack algorithm across all source-target\nclass pairs against this type of neural network model.\n\nEach cell reflects how often inputs from a given\nsource class (x-axis) can be altered into a specific\ntarget class (y-axis) using adversarial perturbations.\nDarker colors indicate higher attack success rates.\n\nUse this matrix to understand how your selected\ninput image and target class affect the overall\nlikelihood of a successful attack.", font=("Arial", 15, "bold")
                                        , justify="center", background="gray", fg="black")
        attack_matrix_label = tk.Label(master = attack_panel, image=photo
                                        , justify="center", background="gray")
        attack_matrix_label.image = photo
        attack_caption_label.pack(pady=10)
        attack_matrix_label.pack(expand=True, fill="y")
        attack_panel.propagate(False)

        #legend_panel = create_panel(root, row=3, column=1)
        # #legend_frame = Frame(root, bd=2, background="gray", relief="sunken")
        # legend_label = Label(legend_panel, text="Green: Source Class         Red: Target Class         Blue: Other Classes", bg="gray", fg="black", font=("Arial", 25, "bold"))
        # legend_label.pack(pady=2)
        # legend_panel.pack(fill="x")
        # legend_panel.propagate(False)

        # valid_digits = ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        # target_class = tk.StringVar()
        # target_class.set(valid_digits[1])

        # target_label = ttk.Label(parameter_panel, text="Target Class:", background="gray")
        # target_label.pack(pady=10, padx=10)
        # target_dropdown = ttk.OptionMenu(parameter_panel, target_class, valid_digits[1], *valid_digits)
        # target_dropdown.pack(pady=5, padx=5)

        # epsilon_label = ttk.Label(parameter_panel, text="Epsilon:", background="gray")
        # epsilon_label.pack(pady=20)
        # epsilon_slider = tk.Scale(parameter_panel, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200)
        # epsilon_slider.pack()

        # # Run the Tkinter main event loop
        # root.mainloop()
    
    def get_input_image(input_panel, method):
        #clear_screen(input_panel)
        nonlocal input_image
        if method == "mnist":
            source_choice_window = tk.Toplevel(root)
            source_choice_window.title("Choose Source Class")            
            source_choice_window.geometry("300x150")

            source_choice = tk.StringVar()
            source_choice.set(valid_digits[0])

            source_choice_frame = Frame(source_choice_window, pady=5, bd=2, background="gray")

            #source_label = ttk.Label(source_choice_frame, text="Source Class:")
            source_label = tk.Label(master = source_choice_frame, text="Source Class:", font=("Arial", 25, "bold")
                                        , justify="center", background="gray", fg="black")
            source_label.pack()
            source_dropdown = OptionMenu(source_choice_frame, source_choice, *valid_digits)
            source_dropdown.config(bg ="gray", fg="black", font=("Arial", 20, "bold"))
            source_dropdown.pack()

            # target_class = tk.StringVar()
            # target_class.set(valid_digits[1])

            # target_label = ttk.Label(source_choice_window, text="Target Class:")
            # target_label.pack(pady=10, padx=10)
            # target_dropdown = ttk.OptionMenu(source_choice_window, target_class, valid_digits[1], *valid_digits)
            # target_dropdown.pack(pady=5, padx=5)

            #done_button = ttk.Button(source_choice_frame, text="Select", command=lambda: get_mnist_input(input_panel, source_choice_window, x_test, y_test, int(source_choice.get())))
            done_button = Button(source_choice_frame, text="Select!", command=lambda: get_mnist_input(input_panel, source_choice_window, x_test, y_test, int(source_choice.get())),
                                     bg="gray", fg="black", font=("Arial", 30, "bold"))
            # done_button = ttk.Button(source_choice_window, text="Select", command=lambda: print("Val: ", source_digit.get()))
            # def on_select():
            #     print("Val:", source_choice.get())

            # done_button = ttk.Button(source_choice_window, text="Select", command=on_select)
            done_button.pack(pady=10)

            source_choice_frame.pack(expand=True, fill="both")
            source_choice_window.mainloop()


            #sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit.get())
        elif method == "draw":
            input_image = get_drawn_digit(convolutional)
            display_input_image(input_panel)


        elif method == "upload":
            file_path = filedialog.askopenfilename()
            input_image = preprocess_uploaded_image(file_path, convolutional)
            display_input_image(input_panel)

        elif method == "camera":
            # Modify to have correct paths to reference
            base_path = None
            digit_path = None

            def start_countdowns():
                # for displaying countdown
                def countdown(t, base, on_complete):
                    def update():
                        nonlocal t
                        if t >= 0:
                            #mins, secs = divmod(t, 60)
                            #secs = 
                            #timer_label.config(text=f"{mins:02d}:{secs:02d}")
                            timer_label.config(text=f"{t}")
                            t -= 1
                            countdown_window.after(1000, update)
                        else:
                            if base:
                                timer_label.config(text="Taking Base Picture Now!")
                                countdown_window.after(1000, on_complete)
                                # PUT CODE HERE: take picture and save to base path
                                # ....................
                                print("Base picture taken!")
                            else:
                                timer_label.config(text="Taking Digit Picture Now!")
                                countdown_window.after(1000, on_complete)
                                # PUT CODE HERE: take picture and save to digit path
                                # ....................
                                print("Digit picture taken!")
                                # do subtraction and save
                                base_img = Image.open(base_path).convert('L')
                                number_img = Image.open(digit_path).convert('L')
                                subtracted_img = ImageChops.subtract(number_img, base_img)
                                subtracted_img.save('subtracted_img.png')

                                # Process image
                                input_image = preprocess_uploaded_image('subtracted_img.png', convolutional)
                                display_input_image(input_panel)
                    update()
                
                # start first countdown
                countdown_window.after(1000, lambda: countdown(3, True, start_second_countdown))

                # to be triggered after completion of first
                def start_second_countdown():
                    timer_label.config(text="Countdown timer for number picture!")
                    countdown(10, False, countdown_window.destroy)
            
            # Create window
            countdown_window = tk.Toplevel()
            countdown_window.title("Countdown Timer")
            countdown_frame = Frame(countdown_window, bd=2, background="gray")
            timer_label = tk.Label(master = countdown_frame, text="Countdown timer for base picture!", font=("Arial", 25, "bold")
                                        , justify="center", background="gray", fg="black")
            #timer_label = tk.Label(countdown_frame, text="Countdown timer for base picture!", font=("Helvetica", 25))
            timer_label.pack(pady=20, expand=True)#, anchor="center")
            countdown_frame.pack(expand=True, fill="both")
            countdown_window.geometry("450x100")

            # start countdowns
            start_countdowns()

            # run window
            countdown_window.mainloop()
        

    def display_input_image(input_panel):
        fig = Figure(figsize=(1,1))
        axes = fig.subplots(1,1)

        clear_screen(input_panel)

        canvas = FigureCanvasTkAgg(fig, master=input_panel)
        canvas.get_tk_widget().pack(expand=True, fill='both')
        canvas.draw()

        axes.set_title(f"Input Image")
        axes.axis('off')
        im = axes.imshow(input_image.reshape(28, 28), cmap='gray')

    def first_load_menu(conv, train):
        nonlocal convolutional, train_model
        convolutional = conv
        train_model = train
        initialize_model()
        load_menu()

    def load_welcome_screen():
        # def on_radio_change():
        #     convolutional = convo_var.get()
        #     train_model = train_var.get()
        
        clear_screen(root)

        welcome_screen_frame = Frame(root, bg="gray", bd=2, relief='sunken')
        welcome_screen_frame.pack(expand=True, fill="both")

        inner_frame = Frame(welcome_screen_frame, bg="gray", bd=2, relief='ridge')
        inner_frame.pack(padx=15, pady=15, expand=True, fill="both")

        space = (tk.Label(inner_frame, text="\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n", bg="gray", fg="black"))
        space.pack()

        welcome_label = tk.Label(inner_frame, text="Welcome to the Adversarial Attack Program!", font=("Arial", 36, "bold", "underline"),
                                 background="gray", fg="black")
        welcome_label.pack(pady=5)

        instruction_label_1 = tk.Label(inner_frame, text="Select from the options below:", font=("Arial", 25, "bold"),
                                       background="gray", fg="black")
        instruction_label_1.pack()

        space = (tk.Label(inner_frame, text=" ", bg="gray", fg="black"))
        space.pack()

        instruction_label_2 = tk.Label(inner_frame, text="Choose type of Deep Neural Network (DNN):", font=("Arial", 18),
                                       background="gray", fg="black")
        instruction_label_2.pack(pady=5)

        convo_var = tk.BooleanVar(value=False)
        radio_mlp = tk.Radiobutton(inner_frame, text="Multi-Layer Perceptron", variable=convo_var, value=False,
                                   background="gray", fg="black")
        radio_mlp.pack()

        radio_convo = tk.Radiobutton(inner_frame, text="Convolutional Neural Network", variable=convo_var, value=True,
                                     background="gray", fg="black")
        radio_convo.pack()

        space = (tk.Label(inner_frame, text=" ", bg="gray", fg="black"))
        space.pack()

        instruction_label_3 = tk.Label(inner_frame, text="Choose whether to load trained model or to train one now:",
                                       font=("Arial", 18), background="gray", fg="black")
        instruction_label_3.pack(pady=5)

        train_var = tk.BooleanVar(value=False)
        radio_load = tk.Radiobutton(inner_frame, text="Load trained model", variable=train_var, value=False,
                                    background="gray", fg="black")
        radio_load.pack()

        radio_train = tk.Radiobutton(inner_frame, text="Train a model now", variable=train_var, value=True,
                                    background="gray", fg="black")
        radio_train.pack()

        done_button = tk.Button(inner_frame, text="Proceed!", font=("Arial", 20, "bold"),
                                command=lambda: first_load_menu(convo_var.get(), train_var.get()))
        done_button.pack(pady=30)


    load_welcome_screen()
    root.mainloop()


if __name__ == "__main__":
    main()
