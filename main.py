import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import io
from train_model import train_model
from evaluate_model import evaluate_model, predict_sample
from adversarial import create_adversarial_example, create_adversarial_example_saliency
from drawing import get_drawn_digit
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 


def get_sample_by_digit(x_test, y_test, source_digit):
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
    return x_test[sample_idx:sample_idx + 1], y_test[sample_idx:sample_idx + 1]


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

def plot_success_rate(success_rates, block=False):
    # Convert success rates to a format that can be plotted
    epsilon_values = np.arange(0.1, 1.1, 0.1)
    top_n_values = range(5, 201, 5)

    success_matrix = np.zeros((len(top_n_values), len(epsilon_values)))

    for epsilon, top_n, distortion, success, adv_probs, adv_image in success_rates:
        epsilon_idx = int((epsilon - 0.1) * 10)
        top_n_idx = (top_n - 5) // 5
        success_matrix[top_n_idx, epsilon_idx] += success

    # Normalize the success rate matrix
    if np.max(success_matrix) > 0:
        success_matrix = success_matrix / np.max(success_matrix)
    else:
        success_matrix = np.zeros_like(success_matrix)

    # Create a new figure with a specific figure number
    plt.figure(num=1, figsize=(10, 6))
    plt.imshow(success_matrix, cmap='YlGnBu', aspect='auto', origin='lower')

    plt.colorbar(label="Success Rate")
    plt.xticks(np.arange(len(epsilon_values)), [f'{epsilon:.1f}' for epsilon in epsilon_values])
    plt.yticks(np.arange(len(top_n_values)), [str(top_n) for top_n in top_n_values])
    plt.xlabel("Epsilon Value")
    plt.ylabel("Top N Pixels Perturbed")
    plt.title("Success Rate of Adversarial Attack Using Saliency Map")
    plt.tight_layout()
    plt.show(block=block)


def plot_min_success_cases(success_rates, model, original_image, original_label, target_label):
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
    if min_epsilon_case is None or min_top_n_case is None:
        print("No successful adversarial examples found.")
        return

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
    fig = plt.figure(num=2, figsize=(20, 20))
    # fig.suptitle("Minimum Successful Adversarial Examples", y=0.95)

    # Case 1: Minimum epsilon successful case
    axes = fig.subplots(3, 4)

    axes[0, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(min_epsilon_adv_image_np.reshape(28, 28), cmap='gray')
    axes[0, 1].set_title(f"Adversarial Image\nMin Epsilon: {round(min_epsilon, 1)}, Top N: {min_epsilon_top_n}")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(perturbation_min_epsilon.reshape(28, 28), cmap='hot')
    axes[0, 2].set_title("Perturbation (Min Epsilon)")
    axes[0, 2].axis('off')

    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    axes[0, 3].bar(digits, min_epsilon_confidence)

    # Case 2: Minimum pixel count successful case
    axes[1, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(min_top_n_adv_image_np.reshape(28, 28), cmap='gray')
    axes[1, 1].set_title(f"Adversarial Image\nMin Top N: {min_top_n}, Epsilon: {round(min_top_n_epsilon, 1)}")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(perturbation_min_top_n.reshape(28, 28), cmap='hot')
    axes[1, 2].set_title("Perturbation (Min Top N)")
    axes[1, 2].axis('off')

    axes[1, 3].bar(digits, min_top_n_confidence)

    # Case 3: Minimum distortion successful case
    axes[2, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
    axes[2, 0].set_title("Original Image")
    axes[2, 0].axis('off')

    axes[2, 1].imshow(min_distortion_adv_image_np.reshape(28, 28), cmap='gray')
    axes[2, 1].set_title(f"Adversarial Image\nMin Distortion: {round(float(min_distortion.numpy()*100), 2)}%, Top N: {round(min_distortion_top_n,1)}, Epsilon: {round(min_distortion_epsilon, 1)}")
    axes[2, 1].axis('off')

    axes[2, 2].imshow(perturbation_min_distortion.reshape(28, 28), cmap='hot')
    axes[2, 2].set_title("Perturbation (Min Top N)")
    axes[2, 2].axis('off')

    axes[2, 3].bar(digits, min_distortion_confidence)

    plt.tight_layout()
    plt.show(block=True)


def clear_screen(root):
   for widget in root.winfo_children():
        widget.destroy()


def main():
    valid_digits = ['', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def specific_test(model, sample_image, sample_label, target_label, epsilon, root):
        print(f"\nGenerating adversarial example for digit: {sample_label} using Saliency map")
        adversarial_image, num_pixels_changed= create_adversarial_example_saliency(root, model, sample_image, sample_label,
                                                                target_label, epsilon, 0, convolutional)

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
        _, adv_probs = predict_sample(model, adversarial_image)
        plot_images(root, sample_image, adversarial_image, difference, adv_probs)



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
            adversarial_image, num_pixels_changed = create_adversarial_example_saliency(root, model, processed_image, original_pred,
                                                                    target_label, epsilon, 150, convolutional)
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
            for n in range(5, 201, 5):
                if n < num_pixels_changed:
                    success_rates.append((epsilon, n, distortion, 0, adv_probs, adversarial_image))
                else:
                    success_rates.append((epsilon, n, distortion, success, adv_probs, adversarial_image))

        # Plot success rate heatmap first, with block=False
        plot_success_rate(success_rates, block=False)

        # Plot minimum success cases second, with block=True
        plot_min_success_cases(success_rates, model, processed_image, np.array([original_pred]),
                                [target_label])

        # Keep both windows open until any key is pressed
        plt.show()
    
    def plot_images(root, original, adversarial, difference, adv_probs):
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
        menu_btn = ttk.Button(root, text="Back to Menu", command=lambda: load_menu())
        menu_btn.pack()

        fig = Figure(figsize=(1,1))
        #fig.patch.set_facecolor("none")
        axes = fig.subplots(2,2)
        canvas = FigureCanvasTkAgg(fig, master = root)
        canvas.draw()
        canvas.get_tk_widget().pack(anchor="center", expand=True, fill="both")

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

        axes[1,1].set_title("Confidence Scores")
        digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        bars = axes[1,1].bar(digits, adv_probs)

        fig.tight_layout()
        # plt.tight_layout()
        # plt.show()
    
    
    # Initialize GUI
    root = tk.Tk()
    root.title("Adversarial Attacks")
    width= root.winfo_screenwidth()               
    height= root.winfo_screenheight()               
    root.geometry("%dx%d" % (width, height))

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
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Model and parameters
    model = None
    convolutional = None
    train_model = None
    
    def initialize_model():
        # Parameters
        # convolutional = False    # True: CNN, False: MLP
        # train_model = False       # True: training model, False: loading model
        nonlocal x_train, x_test, convolutional, train_model, model

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

        print("Model baseline accuracy: ", model.evaluate(x_test, y_test)[1]) 
        # Show model architecture
        # model.summary()

    
    def test_mnist_iterate():
        def mnist_iterate(source_digit, target_label):
            try:
                # Get a sample image of the requested digit
                sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)
                iteration_test(model, sample_image, target_label, root)
            except ValueError as e:
                print(f"Error: {e}")

        # new_root.mainloop()
        clear_screen(root)
    
        # generate new frame
        test_mnist_iterate_frame = ttk.Frame(root, padding=10)
        test_mnist_iterate_frame.pack(expand=True, fill="x")

        # instruction label 1
        instruction_label_1 = tk.Label(master = test_mnist_iterate_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_1.pack()

        # source class entry
        source_digit = tk.StringVar()
        source_digit.set(valid_digits[0])

        # target class entry
        target_class = tk.StringVar()
        target_class.set(valid_digits[1])

        source_label = ttk.Label(test_mnist_iterate_frame, text="Source Class:")
        source_label.pack()
        source_dropdown = ttk.OptionMenu(test_mnist_iterate_frame, source_digit, *valid_digits)
        source_dropdown.pack()

        target_label = ttk.Label(test_mnist_iterate_frame, text="Target Class:")
        target_label.pack()
        target_dropdown = ttk.OptionMenu(test_mnist_iterate_frame, target_class, *valid_digits)
        target_dropdown.pack()

        # Attack button
        button = ttk.Button(master=test_mnist_iterate_frame, text="Run Attack!",
                                command=lambda: mnist_iterate(int(source_digit.get()),
                                                            int(target_class.get())))
        button.pack(pady=5)

        # menu button
        menu_btn = ttk.Button(test_mnist_iterate_frame, text="Back to Menu", command=lambda: load_menu())
        menu_btn.pack(pady = 50)


    def test_uploaded_iterate():
        def file_option(target_label):
            # get image from file system
            file_path = filedialog.askopenfilename()
            try:
                processed_image = preprocess_uploaded_image(file_path, convolutional)
                iteration_test(model, processed_image, target_label, root)
            except Exception as e:
                print(f"Error processing image: {str(e)}")   

        def draw_option(target_label):
            # get image from drawing
            try:
                processed_image = get_drawn_digit()
                iteration_test(model, processed_image, target_label, root)
            except Exception as e:
                print(f"Error processing image: {str(e)}")

        def picture_option(target_label):
            # Test with uploaded image or drawing
            try:
                processed_image = preprocess_uploaded_image('/home/designteam10/Pictures/image.jpg')
                iteration_test(model, processed_image, target_label, root)
            except Exception as e:
                print(f"Error processing image: {str(e)}")


        clear_screen(root)

        # generate new frame
        test_uploaded_iterate_frame = ttk.Frame(root, padding=10)
        test_uploaded_iterate_frame.pack(expand=True, fill="x")

        # instruction label 1
        instruction_label_1 = tk.Label(master = test_uploaded_iterate_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_1.pack()

        # target class entry
        target_class = tk.StringVar()
        target_class.set(valid_digits[1])

        input_frame1 = ttk.Frame(master=test_uploaded_iterate_frame)
        target_label = ttk.Label(test_uploaded_iterate_frame, text="Target Class:")
        target_label.pack()
        target_dropdown = ttk.OptionMenu(test_uploaded_iterate_frame, target_class, *valid_digits)
        target_dropdown.pack()
        input_frame1.pack(pady=5)

        # Instruction label 2
        instruction_label_2 = tk.Label(master = test_uploaded_iterate_frame, text="Choose one of the options below to run the attack:", font=("Arial", 14), justify="center")
        instruction_label_2.pack()

        # Create upload button
        upload_btn = ttk.Button(test_uploaded_iterate_frame, text="Upload Image", command=lambda: file_option(int(target_class.get())))
        upload_btn.pack(pady=5)

        # Draw digit button
        draw_btn = ttk.Button(test_uploaded_iterate_frame, text="Draw Digit", command=lambda: draw_option(int(target_class.get())))
        draw_btn.pack(pady=5)

        # Take picture button
        picture_btn = ttk.Button(test_uploaded_iterate_frame, text="Take Picture", command=lambda: picture_option(int(target_class.get())))
        picture_btn.pack(pady=5)

        # menu button
        menu_btn = ttk.Button(test_uploaded_iterate_frame, text="Back to Menu", command=lambda: load_menu())
        menu_btn.pack(pady = 50)



    def test_mnist_specify():
        def mnist_specify(source_digit, target_label, epsilon):
            try:
                # Get a sample image of the requested digit
                sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)
                specific_test(model, sample_image, sample_label[0], target_label, epsilon, root)
            except ValueError as e:
                print(f"Error: {e}")

        clear_screen(root)
    
        # generate new frame
        test_mnist_specify_frame = ttk.Frame(root, padding=10)
        test_mnist_specify_frame.pack(expand=True, fill="x")

        # instruction label 1
        instruction_label_1 = tk.Label(master = test_mnist_specify_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_1.pack()

        # source class entry
        source_digit = tk.StringVar()
        source_digit.set(valid_digits[0])

        # target class entry
        target_class = tk.StringVar()
        target_class.set(valid_digits[1])

        source_label = ttk.Label(test_mnist_specify_frame, text="Source Class:")
        source_label.pack()
        source_dropdown = ttk.OptionMenu(test_mnist_specify_frame, source_digit, *valid_digits)
        source_dropdown.pack()

        target_label = ttk.Label(test_mnist_specify_frame, text="Target Class:")
        target_label.pack()
        target_dropdown = ttk.OptionMenu(test_mnist_specify_frame, target_class, *valid_digits)
        target_dropdown.pack()


        # epsilon entry
        input_frame3 = ttk.Frame(master=test_mnist_specify_frame)
        epsilon_label = ttk.Label(input_frame3, text="Epsilon:")
        epsilon_label.pack()
        epsilon_slider = tk.Scale(input_frame3, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200)
        epsilon_slider.pack()
        input_frame3.pack(pady=5)

        # Attack button
        button = ttk.Button(master=test_mnist_specify_frame, text="Run Attack!",
                                command=lambda: mnist_specify(int(source_digit.get()),
                                                            int(target_class.get()),
                                                            float(epsilon_slider.get())))
        button.pack(pady=10)

        # menu button
        menu_btn = ttk.Button(test_mnist_specify_frame, text="Back to Menu", command=lambda: load_menu())
        menu_btn.pack(pady = 50)


    def test_uploaded_specify():
        def file_option(target_label, epsilon):
            file_path = filedialog.askopenfilename()
            try:
                processed_image = preprocess_uploaded_image(file_path, convolutional)
                pred, probs = predict_sample(model, processed_image)
                specific_test(model, processed_image, pred, target_label, epsilon, root)
            except Exception as e:
                print(f"Error processing image: {str(e)}")

        def draw_option(target_label, epsilon):
            try:
                processed_image = get_drawn_digit()
                pred, probs = predict_sample(model, processed_image)
                specific_test(model, processed_image, pred, target_label, epsilon, root)
            except Exception as e:
                print(f"Error processing image: {str(e)}")

        def picture_option(target_label, epsilon):
            try:
                processed_image = preprocess_uploaded_image('/home/designteam10/Pictures/image.jpg')
                #print(processed_image)
                pred, probs = predict_sample(model, processed_image)
                specific_test(model, processed_image, pred, target_label, epsilon, root)
            except Exception as e:
                print(f"Error processing image: {str(e)}")

        clear_screen(root)
    
        # generate new frame
        test_uploaded_specify_frame = ttk.Frame(root, padding=10)
        test_uploaded_specify_frame.pack(expand=True, fill="x")

        # instruction label 1
        instruction_label_1 = tk.Label(master = test_uploaded_specify_frame, text="Enter in attack parameters:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_1.pack()

        # epsilon entry
        input_frame1 = ttk.Frame(master=test_uploaded_specify_frame)
        epsilon_label = ttk.Label(input_frame1, text="Epsilon:")
        epsilon_label.pack()
        epsilon_slider = tk.Scale(input_frame1, from_=0.1, to=1.0, resolution=0.01, orient=HORIZONTAL, length=200)
        epsilon_slider.pack(padx=10)
        input_frame1.pack(pady=5)

        # target class entry
        target_class = tk.StringVar()
        target_class.set(valid_digits[1])

        input_frame3 = ttk.Frame(master=test_uploaded_specify_frame)
        target_label = ttk.Label(input_frame3, text="Target Class:")
        target_label.pack()
        target_dropdown = ttk.OptionMenu(input_frame3, target_class, *valid_digits)
        target_dropdown.pack()
        input_frame3.pack(pady=5)

        # Instruction label 2
        instruction_label_2 = tk.Label(master = test_uploaded_specify_frame, text="Choose one of the options below to run the attack:", font=("Arial", 14), justify="center")
        instruction_label_2.pack()

        # Create upload button
        upload_btn = ttk.Button(test_uploaded_specify_frame, text="Upload Image", command=lambda: file_option(int(target_class.get()),
                                                                                            float(epsilon_slider.get())))
        upload_btn.pack(pady=5)

        # Draw digit button
        draw_btn = ttk.Button(test_uploaded_specify_frame, text="Draw Digit", command=lambda: draw_option(int(target_class.get()),
                                                                                        float(epsilon_slider.get())))
        draw_btn.pack(pady=5)

        # Take picture button
        picture_btn = ttk.Button(test_uploaded_specify_frame, text="Take Picture", command=lambda: picture_option(int(target_class.get()),
                                                                                        float(epsilon_slider.get())))
        picture_btn.pack(pady=5)

        # menu button
        #menu_frame = ttk.Frame(master=test_uploaded_specify_frame)
        menu_btn = ttk.Button(test_uploaded_specify_frame, text="Back to Menu", command=lambda: load_menu())
        menu_btn.pack(pady = 50)
        # menu_frame.pack(expand=True,fill="x")

    # Option screen
    def load_menu():
        clear_screen(root)
        option_screen_frame = ttk.Frame(root, padding=10)
        option_screen_frame.pack(expand=True, fill="x")
        option_label = tk.Label(master = option_screen_frame, text="Choose an option:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        option_label.pack()
        button1 = ttk.Button(master = option_screen_frame, text="Test MNIST digits across multiple epsilons", command=test_mnist_iterate)#.grid(column=0, row=1)
        button1.pack(padx=20, pady=5)
        button2 = ttk.Button(master = option_screen_frame, text="Test uploaded image across multiple epsilons", command=test_uploaded_iterate)#.grid(column=0, row=2)
        button2.pack(padx=20, pady=5)
        button3 = ttk.Button(master = option_screen_frame, text="Create an adversarial example for random number from MNIST dataset (specify max epsilon)", command=test_mnist_specify)#.grid(column=0, row=3)
        button3.pack(padx=20, pady=5)
        button4 = ttk.Button(master = option_screen_frame, text="Create an adversarial example for uploaded picture (specify max epsilon)", command=test_uploaded_specify)#.grid(column=0, row=4)
        button4.pack(padx=20, pady=5)
        tk.Label(master = option_screen_frame, text=" ", font=("Arial", 20), justify="center").pack()
        button5 = ttk.Button(master = option_screen_frame, text="Choose new neural network model", command=lambda: load_welcome_screen())
        button5.pack()
        tk.Label(master = option_screen_frame, text=" ", font=("Arial", 6), justify="center").pack()
        button6 = ttk.Button(master = option_screen_frame, text="Exit program", command=root.destroy)
        button6.pack()

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
        welcome_screen_frame = ttk.Frame(root, padding=10)
        welcome_screen_frame.pack(expand=True, fill="x")
        welcome_label = tk.Label(master = welcome_screen_frame, text="Welcome to the Adversarial Attack Program!", font=("Arial", 14), justify="center")
        welcome_label.pack(pady=5)
        instruction_label_1 = tk.Label(master = welcome_screen_frame, text="Select from the options below:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_1.pack()
        tk.Label(master = welcome_screen_frame, text=" ", font=("Arial", 14), justify="center").pack()
        instruction_label_2 = tk.Label(master = welcome_screen_frame, text="Choose type of Deep Neural Network (DNN):", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_2.pack(pady=5)
        convo_var = tk.BooleanVar(value = False)
        radio_mlp = ttk.Radiobutton(welcome_screen_frame, text="Multi-Layer Perceptron", variable=convo_var, value=False)#, command=on_radio_change)
        radio_mlp.pack()
        radio_convo = ttk.Radiobutton(welcome_screen_frame, text="Convolutional Neural Network", variable=convo_var, value=True)#, command=on_radio_change)
        radio_convo.pack()
        tk.Label(master = welcome_screen_frame, text=" ", font=("Arial", 14), justify="center").pack()
        instruction_label_3 = tk.Label(master = welcome_screen_frame, text="Choose whether to load trained model or to train one now:", font=("Arial", 14), justify="center")#.grid(column=0, row=0)
        instruction_label_3.pack(pady=5)
        train_var = tk.BooleanVar(value = False)
        radio_load = ttk.Radiobutton(welcome_screen_frame, text="Load trained model", variable=train_var, value=False)#, command=on_radio_change)
        radio_load.pack()
        radio_train = ttk.Radiobutton(welcome_screen_frame, text="Train a model now", variable=train_var, value=True)#, command=on_radio_change)
        radio_train.pack()
        done_button = ttk.Button(master = welcome_screen_frame, text="Proceed!", command=lambda: first_load_menu(convo_var.get(), train_var.get()))
        done_button.pack(pady=30)


        

    load_welcome_screen()
    root.mainloop()


if __name__ == "__main__":
    main()