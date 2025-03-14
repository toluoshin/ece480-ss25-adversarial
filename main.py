import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
import io
from train_model import train_model
from evaluate_model import evaluate_model, predict_sample
from adversarial import create_adversarial_example, create_adversarial_example_saliency
from drawing import get_drawn_digit


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


def preprocess_uploaded_image(image_path):
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


def plot_images(original, adversarial, difference):
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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    im1 = axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])

    # Plot adversarial image
    im2 = axes[1].imshow(adversarial, cmap='gray')
    axes[1].set_title("Adversarial Image")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])

    # Plot the difference (noise)
    # Normalize difference to symmetric range for seismic colormap
    max_diff = max(abs(difference.max()), abs(difference.min()))
    im3 = axes[2].imshow(difference, cmap='seismic', vmin=-max_diff, vmax=max_diff)
    axes[2].set_title("Perturbation (Difference)")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()

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
    plot_confidence_text(min_epsilon_top_3_data, min_top_n_top_3_data, min_distortion_top_3_data)

    # Create a new figure with a specific figure number
    fig = plt.figure(num=2, figsize=(20, 20))
    # fig.suptitle("Minimum Successful Adversarial Examples", y=0.95)

    # Case 1: Minimum epsilon successful case
    axes = fig.subplots(3, 3)

    axes[0, 0].imshow(original_image_np.reshape(28, 28), cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(min_epsilon_adv_image_np.reshape(28, 28), cmap='gray')
    axes[0, 1].set_title(f"Adversarial Image\nMin Epsilon: {round(min_epsilon, 1)}, Top N: {min_epsilon_top_n}")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(perturbation_min_epsilon.reshape(28, 28), cmap='hot')
    axes[0, 2].set_title("Perturbation (Min Epsilon)")
    axes[0, 2].axis('off')

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

    plt.tight_layout()
    plt.show(block=True)


def interactive_adversarial_demo():
    # Load and preprocess MNIST dataset for training
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Train the model
    print("Training the model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    while True:
        print("\nChoose an option:")
        print("1. Test MNIST digits with Saliency map (iterate epsilon and top N)")
        print("2. Test uploaded image with Saliency map (iterate epsilon and top N)")
        print("3. Test MNIST digits with Saliency map (specify epsilon and top N)")
        print("4. Test uploaded image with Saliency map (specify epsilon and top N)")
        print("5. Quit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            try:
                # get the requested source digit
                source_digit = int(input("\nEnter a digit (0-9) to generate an adversarial example for: "))
                if source_digit < 0 or source_digit > 9:
                    print("Please enter a valid digit between 0 and 9")
                    continue

                # get the rqeuested target class
                target_label = int(input("\nEnter a digit (0-9) to force misclassification to: "))
                if target_label < 0 or target_label > 9:
                    print("Please enter a valid digit between 0 and 9")
                    continue
                
                # Get a sample image of the requested digit
                sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)

                original_pred, original_probs = predict_sample(model, sample_image)
                print("\nOriginal Prediction:")
                print(f"Predicted digit: {original_pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(original_probs)[-3:][::-1]
                for digit, prob in zip(top3, original_probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Iterate over epsilon and top N values
                success_rates = []
                for epsilon in np.arange(0.1, 1.1, 0.1):
                    for top_n in range(5, 201, 5):
                        print(f"Testing epsilon={epsilon:.1f}, Top N={top_n}")
                        adversarial_image = create_adversarial_example_saliency(model, sample_image, sample_label[0],
                                                                                target_label, epsilon, top_n)

                        # Get prediction for adversarial image
                        original_pred, original_probs = predict_sample(model, sample_image)
                        adv_pred, adv_probs = predict_sample(model, adversarial_image)

                        success = 1 if adv_pred == target_label else 0
                        success_rates.append((epsilon, top_n, success))
            
                # Plot success rate heatmap first, with block=False
                plot_success_rate(success_rates, block=False)

                # Plot minimum success cases second, with block=True
                plot_min_success_cases(success_rates, model, sample_image, sample_label, target_label)

                # Keep both windows open until any key is pressed
                plt.show()

            except ValueError as e:
                print(f"Error: {e}")
                continue

        elif choice == '2':
            # Test with uploaded image or drawing
            input_method = input("\nSelect input method:\n1. Upload image\n2. Draw digit\nEnter choice (1-2): ")
            try:
                if input_method == '1':
                    image_path = input("\nEnter the path to your image file: ")
                    processed_image = preprocess_uploaded_image(image_path)
                elif input_method == '2':
                    print("\nDrawing window will open. Draw your digit and click 'Save & Process' when done.")
                    processed_image = get_drawn_digit()
                else:
                    print("Invalid choice.")
                    continue
                
                # get the rqeuested target class
                target_label = int(input("\nEnter a digit (0-9) to force misclassification to: "))
                if target_label < 0 or target_label > 9:
                    print("Please enter a valid digit between 0 and 9")
                    continue

                # Get prediction for original image
                original_pred, original_probs = predict_sample(model, processed_image)
                print("\nOriginal Prediction:")
                print(f"Predicted digit: {original_pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(original_probs)[-3:][::-1]
                for digit, prob in zip(top3, original_probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Iterate over epsilon and top N values
                success_rates = []
                for epsilon in np.arange(0.1, 1.1, 0.1):
                    for top_n in range(5, 201, 5):
                        print(f"Testing epsilon={epsilon:.1f}, Top N={top_n}")
                        adversarial_image = create_adversarial_example_saliency(model, processed_image,
                                                                                original_pred, target_label, epsilon,
                                                                                top_n)

                        # Get prediction for adversarial image
                        adv_pred, adv_probs = predict_sample(model, adversarial_image)

                        #success = 1 if adv_pred != original_pred else 0
                        success = 1 if adv_pred == target_label else 0
                        distortion = tf.reduce_sum(abs(adversarial_image - processed_image))
                        print(distortion)
                        success_rates.append((epsilon, top_n, success))

                # Plot success rate heatmap first, with block=False
                plot_success_rate(success_rates, block=False)

                # Plot minimum success cases second, with block=True
                plot_min_success_cases(success_rates, model, processed_image, np.array([original_pred]), [target_label])

                # Keep both windows open until any key is pressed
                plt.show()

            except Exception as e:
                print(f"Error processing image: {str(e)}")

        elif choice == '3':
            # Select source digit and epsilon value
            try:
                source_digit = int(input("\nEnter a digit (0-9) to generate an adversarial example for: "))
                if source_digit < 0 or source_digit > 9:
                    print("Please enter a valid digit between 0 and 9")
                    continue

                # get the rqeuested target class
                target_label = int(input("\nEnter a digit (0-9) to force misclassification to: "))
                if target_label < 0 or target_label > 9:
                    print("Please enter a valid digit between 0 and 9")
                    continue

                # Get a sample image of the requested digit
                sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)

                # Set epsilon value
                epsilon = float(input("Enter epsilon value for perturbation (0.1-0.3 recommended): "))

                # Saliency map
                top_n = int(input("Enter the number of top N pixels to perturb: "))
                print(f"\nGenerating adversarial example for digit: {source_digit} using Saliency map")
                adversarial_image = create_adversarial_example_saliency(model, sample_image[0], sample_label, target_label, epsilon,
                                                                        top_n)

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
                    print(f"\nSuccess! Model was fooled: {source_digit} → {target_label}")
                else:
                    print("\nThe model wasn't fooled. Try increasing epsilon or top N.")

                # Visualize the results
                difference = adversarial_image - sample_image
                plot_images(sample_image, adversarial_image, difference)

            except ValueError as e:
                print(f"Error: {e}")
                continue


        elif choice == '4':
            input_method = input("\nSelect input method:\n1. Upload image\n2. Draw digit\nEnter choice (1-2): ")
            try:
                if input_method == '1':
                    image_path = input("\nEnter the path to your image file: ")
                    processed_image = preprocess_uploaded_image(image_path)
                elif input_method == '2':
                    print("\nDrawing window will open. Draw your digit and click 'Save & Process' when done.")
                    processed_image = get_drawn_digit()
                else:
                    print("Invalid choice.")
                    continue
                
                # get the rqeuested target class
                target_label = int(input("\nEnter a digit (0-9) to force misclassification to: "))
                if target_label < 0 or target_label > 9:
                    print("Please enter a valid digit between 0 and 9")
                    continue

                # Get prediction for original image
                pred, probs = predict_sample(model, processed_image)

                print("\nPrediction for input image:")
                print(f"Predicted digit: {pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(probs)[-3:][::-1]
                for digit, prob in zip(top3, probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Set epsilon value
                epsilon = float(input("\nEnter epsilon value for adversarial perturbation (0.1-0.3 recommended): "))

                # Saliency Map
                top_n = int(input("Enter the number of top N pixels to perturb: "))
                adversarial_image = create_adversarial_example_saliency(model, processed_image, pred,
                                                                        target_label, epsilon, top_n)

                # Get prediction for adversarial image
                adv_pred, adv_probs = predict_sample(model, adversarial_image)

                print("\nPrediction for adversarial image:")
                print(f"Predicted digit: {adv_pred}")
                print("\nTop 3 probabilities:")
                top3_adv = np.argsort(adv_probs)[-3:][::-1]
                for digit, prob in zip(top3_adv, adv_probs[top3_adv]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Plot both images
                plot_uploaded_comparison(processed_image, adversarial_image)

            except Exception as e:
                print(f"Error processing image: {str(e)}")
        elif choice == '5':
            print("Goodbye!")
            break

        else:
            print("Invalid choice. Please try again.")

def main():
    # Load and preprocess MNIST dataset for training
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize and reshape data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    # Train the model
    print("Training the model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

    root = tk.Tk()
    root.title("Adversarial Attacks")
    root.geometry("700x300")  # Width x Height

    label = ttk.Label(master = root, text="Choose an option:", font=("Arial", 14))
    label.pack()

    def test_mnist_iterate():
        new_root = tk.Tk()
        new_root.title("Choose a Digit")
        new_root.geometry("300x200")

        new_label = ttk.Label(new_root, text="Enter a Digit 0-9", font=("Arial", 14))
        new_label.pack(pady=20)

        def mnist_iterate(source_digit, target_label):
            try:
                # Get a sample image of the requested digit
                sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)

                original_pred, original_probs = predict_sample(model, sample_image)
                print("\nOriginal Prediction:")
                print(f"Predicted digit: {original_pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(original_probs)[-3:][::-1]
                for digit, prob in zip(top3, original_probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Iterate over epsilon and top N values
                success_rates = []

                # for i in range(2):
                #     adversarial_image = create_adversarial_example_saliency(model, sample_image, sample_label[0],
                #                                                             target_label, 0.5, 30)
                #     adv_pred, adv_probs = predict_sample(model, adversarial_image)

                #     print("adv_pred: ", adv_pred)
                #     print("adv_probs: ", adv_probs)

                #     if tf.is_tensor(sample_image):
                #         sample_image = sample_image.numpy()

                #     # Create subplots based on whether adversarial image is provided
                #     n_plots = 3 if adversarial_image is not None else 1
                #     fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))  # ✅ Correctly creates figure

                #     if n_plots == 1:
                #         axes = [axes]  # Make axes iterable for single plot

                #     # Plot uploaded image
                #     sample_display = sample_image.reshape(28, 28)
                #     im1 = axes[0].imshow(sample_display, cmap='gray')
                #     axes[0].set_title("Sample Image")
                #     axes[0].axis('off')
                #     plt.colorbar(im1, ax=axes[0])

                #     if adversarial_image is not None:
                #         if tf.is_tensor(adversarial_image):
                #             adversarial_image = adversarial_image.numpy()

                #         # Plot adversarial image
                #         adversarial_display = adversarial_image.reshape(28, 28)
                #         im2 = axes[1].imshow(adversarial_display, cmap='gray')
                #         axes[1].set_title("Adversarial Image")
                #         axes[1].axis('off')
                #         plt.colorbar(im2, ax=axes[1])

                #         # Plot difference
                #         difference = adversarial_image - sample_image
                #         difference_display = difference.reshape(28, 28)
                #         max_diff = max(abs(difference_display.max()), abs(difference_display.min()))
                #         im3 = axes[2].imshow(difference_display, cmap='seismic', vmin=-max_diff, vmax=max_diff)
                #         axes[2].set_title("Perturbation (Difference)")
                #         axes[2].axis('off')
                #         plt.colorbar(im3, ax=axes[2])

                #     plt.tight_layout()
                #     plt.show(block=False)  # ✅ Prevents blocking issues in loops
                #     #plt.pause(1)  # ✅ Allows time for rendering in interactive mode
                #     plt.close(fig)  # ✅ Frees memory for the next iteration

                # for i in range(2):
                    
                #     adversarial_image = create_adversarial_example_saliency(model, sample_image, sample_label[0],
                #                                                                     target_label, 0.5, 30)
                #     adv_pred, adv_probs = predict_sample(model, adversarial_image)

                #     print("adv_pred: ", adv_pred)
                #     print("adv_probs: ", adv_probs)
                #     if tf.is_tensor(sample_image):
                #         sample_image = sample_image.numpy()

                #     # Create subplots based on whether adversarial image is provided
                #     n_plots = 3 if adversarial_image is not None else 1
                #     fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

                #     if n_plots == 1:
                #         axes = [axes]  # Make axes iterable for single plot

                #     # Plot uploaded image
                #     sample_display = sample_image.reshape(28, 28)
                #     im1 = axes[0].imshow(sample_display, cmap='gray')
                #     axes[0].set_title("Sample Image")
                #     axes[0].axis('off')
                #     plt.colorbar(im1, ax=axes[0])

                #     if adversarial_image is not None:
                #         if tf.is_tensor(adversarial_image):
                #             adversarial_image = adversarial_image.numpy()

                #         # Plot adversarial image
                #         adversarial_display = adversarial_image.reshape(28, 28)
                #         im2 = axes[1].imshow(adversarial_display, cmap='gray')
                #         axes[1].set_title("Adversarial Image")
                #         axes[1].axis('off')
                #         plt.colorbar(im2, ax=axes[1])

                #         # Plot difference
                #         difference = adversarial_image - sample_image
                #         difference_display = difference.reshape(28, 28)
                #         max_diff = max(abs(difference_display.max()), abs(difference_display.min()))
                #         im3 = axes[2].imshow(difference_display, cmap='seismic', vmin=-max_diff, vmax=max_diff)
                #         axes[2].set_title("Perturbation (Difference)")
                #         axes[2].axis('off')
                #         plt.colorbar(im3, ax=axes[2])

                #     plt.figure()
                #     plt.tight_layout()
                #     plt.show()

                    

                for epsilon in np.arange(0.1, 1.1, 0.1):
                    #print(f"Testing epsilon={epsilon:.1f}, Top N={top_n}")
                    print(f"Testing epsilon={epsilon:.1f}")
                    adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, sample_image, sample_label[0],
                                                                            target_label, epsilon, 150)
                    # Get prediction for adversarial image
                    original_pred, original_probs = predict_sample(model, sample_image)
                    adv_pred, adv_probs = predict_sample(model, adversarial_image)

                    success = 1 if adv_pred == target_label else 0
                    distortion = 0
                    if success:
                        print("Adversarial example can be made with epsilon ", epsilon, ", ", num_pixels_changed, " had to be perturbed.")
                        distortion = tf.reduce_sum(abs(sample_image - adversarial_image)) / float(784)
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
                plot_min_success_cases(success_rates, model, sample_image, sample_label, [target_label])

                # Keep both windows open until any key is pressed
                plt.show()

            except ValueError as e:
                print(f"Error: {e}")

            new_root.destroy()

        # Source digit input
        input_frame1 = ttk.Frame(master=new_root)
        source_label = ttk.Label(input_frame1, text="Source Digit:")
        source_entry = ttk.Entry(input_frame1, width=5)
        source_label.pack(side='left', padx=5)
        source_entry.pack(side='left', padx=10)
        input_frame1.pack(pady=5)

        # Target classification digit input
        input_frame2 = ttk.Frame(master=new_root)
        target_label = ttk.Label(input_frame2, text="Target Class:")
        target_entry = ttk.Entry(input_frame2, width=5)
        target_label.pack(side='left', padx=5)
        target_entry.pack(side='left', padx=10)
        input_frame2.pack(pady=5)

        # Single button for both entries
        button = ttk.Button(master=new_root, text="Test Digits", command=lambda: mnist_iterate(int(source_entry.get()), int(target_entry.get())))
        button.pack(pady=10)

        new_root.mainloop()


    def test_uploaded_iterate():
        new_root = tk.Tk()
        new_root.title("Upload or Draw Image")
        new_root.geometry("300x300")

        def upload_image(target_label):
            file_path = filedialog.askopenfilename()
            if file_path:
                # Test with uploaded image or drawing
                try:
                    processed_image = preprocess_uploaded_image(file_path)


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
                        adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, processed_image, original_pred,
                                                                                target_label, epsilon, 150)
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

                except Exception as e:
                    print(f"Error processing image: {str(e)}")

                new_root.destroy()

        def draw_digit(target_label):
            # Test with uploaded image or drawing
            try:
                processed_image = get_drawn_digit()

                # Get prediction for original image
                original_pred, original_probs = predict_sample(model, processed_image)
                print("\nOriginal Prediction:")
                print(f"Predicted digit: {original_pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(original_probs)[-3:][::-1]
                for digit, prob in zip(top3, original_probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Iterate over epsilon and top N values
                success_rates = []
                for epsilon in np.arange(0.1, 1.1, 0.1):
                    #print(f"Testing epsilon={epsilon:.1f}, Top N={top_n}")
                    print(f"Testing epsilon={epsilon:.1f}")
                    adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, processed_image, original_pred,
                                                                            target_label, epsilon, 150)
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

            except Exception as e:
                print(f"Error processing image: {str(e)}")

            new_root.destroy()

        def picture(target_label):
            # Test with uploaded image or drawing
            try:
                processed_image = preprocess_uploaded_image('/home/designteam10/Pictures/image.jpg')

                # Get prediction for original image
                original_pred, original_probs = predict_sample(model, processed_image)
                print("\nOriginal Prediction:")
                print(f"Predicted digit: {original_pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(original_probs)[-3:][::-1]
                for digit, prob in zip(top3, original_probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Iterate over epsilon and top N values
                success_rates = []
                for epsilon in np.arange(0.1, 1.1, 0.1):
                    #print(f"Testing epsilon={epsilon:.1f}, Top N={top_n}")
                    print(f"Testing epsilon={epsilon:.1f}")
                    adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, processed_image, original_pred,
                                                                            target_label, epsilon, 150)
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

            except Exception as e:
                print(f"Error processing image: {str(e)}")

            new_root.destroy()

        input_frame = ttk.Frame(master=new_root)
        target_label = ttk.Label(input_frame, text="Target Class:")
        target_entry = ttk.Entry(input_frame, width=5)
        target_label.pack(side='left', padx=5)
        target_entry.pack(side='left', padx=10)
        input_frame.pack(pady=10)

        # Create upload button
        upload_btn = ttk.Button(new_root, text="Upload Image", command=lambda: upload_image(int(target_entry.get())))
        upload_btn.pack(pady=15)

        # Draw digit button
        draw_btn = ttk.Button(new_root, text="Draw Digit", command=lambda: draw_digit(int(target_entry.get())))
        draw_btn.pack(pady=20)

        # Picture Button
        pciture_btn = ttk.Button(new_root, text="Take Picture", command=lambda: picture(int(target_entry.get())))
        pciture_btn.pack(pady=25)

        # Label to display the image
        img_label = ttk.Label(new_root)
        img_label.pack()

        # Run the application
        new_root.mainloop()

    def test_mnist_specify():
        new_root = tk.Tk()
        new_root.title("Choose a Digit")
        new_root.geometry("300x350")

        new_label = ttk.Label(new_root, text="Enter a Digit 0-9", font=("Arial", 14))
        new_label.pack(pady=20)

        def mnist_specify(source_digit, target_label, epsilon):
            # Select source digit and epsilon value
            try:
                # Get a sample image of the requested digit
                sample_image, sample_label = get_sample_by_digit(x_test, y_test, source_digit)

                # Saliency map
                print(f"\nGenerating adversarial example for digit: {source_digit} using Saliency map")
                adversarial_image, num_pixels_changed= create_adversarial_example_saliency(model, sample_image, sample_label[0],
                                                                        target_label, epsilon,
                                                                        0)

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
                    print(f"\nSuccess! Model was fooled: {source_digit} → {target_label}")
                    distortion = tf.reduce_sum(abs(sample_image - adversarial_image)) / float(784)
                    print("Input feature distortion: ", round(float(distortion.numpy()*100), 2), "%")
                else:
                    print("\nThe model wasn't fooled. Try increasing epsilon or top N.")

                # Visualize the results
                difference = adversarial_image - sample_image
                plot_images(sample_image, adversarial_image, difference)

            except ValueError as e:
                print(f"Error: {e}")

            new_root.destroy()

        # Source digit input
        input_frame1 = ttk.Frame(master=new_root)
        source_label = ttk.Label(input_frame1, text="Source Digit:")
        source_entry = ttk.Entry(input_frame1, width=5)
        source_label.pack(side='left', padx=5)
        source_entry.pack(side='left', padx=10)
        input_frame1.pack(pady=5)

        # Target classification digit input
        input_frame2 = ttk.Frame(master=new_root)
        target_label = ttk.Label(input_frame2, text="Target Class:")
        target_entry = ttk.Entry(input_frame2, width=5)
        target_label.pack(side='left', padx=5)
        target_entry.pack(side='left', padx=10)
        input_frame2.pack(pady=5)

        # Epsilon input
        input_frame3 = ttk.Frame(master=new_root)
        epsilon_label = ttk.Label(input_frame3, text="Epsilon:")
        epsilon_entry = ttk.Entry(input_frame3, width=5)
        epsilon_label.pack(side='left', padx=5)
        epsilon_entry.pack(side='left', padx=10)
        input_frame3.pack(pady=5)

        # Top N input
        # input_frame4 = ttk.Frame(master=new_root)
        # topn_label = ttk.Label(input_frame4, text="Top N:")
        # topn_entry = ttk.Entry(input_frame4, width=5)
        # topn_label.pack(side='left', padx=5)
        # topn_entry.pack(side='left', padx=10)
        # input_frame4.pack(pady=5)

        # Single button for both entries
        button = ttk.Button(master=new_root, text="Test Digits",
                            command=lambda: mnist_specify(int(source_entry.get()),
                                                          int(target_entry.get()),
                                                          float(epsilon_entry.get())))
        button.pack(pady=10)

        new_root.mainloop()

    def test_uploaded_specify():
        new_root = tk.Tk()
        new_root.title("Upload or Draw Image")
        new_root.geometry("300x400")

        def upload_image(target_label, epsilon):
            file_path = filedialog.askopenfilename()
            if file_path:
                try:
                    processed_image = preprocess_uploaded_image(file_path)

                    # Get prediction for original image
                    pred, probs = predict_sample(model, processed_image)

                    print("\nPrediction for input image:")
                    print(f"Predicted digit: {pred}")
                    print("\nTop 3 probabilities:")
                    top3 = np.argsort(probs)[-3:][::-1]
                    for digit, prob in zip(top3, probs[top3]):
                        print(f"Digit {digit}: {prob * 100:.2f}%")

                    # Saliency Map
                    adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, processed_image, pred,
                                                                            target_label, epsilon, 0)

                    # Get prediction for adversarial image
                    adv_pred, adv_probs = predict_sample(model, adversarial_image)

                    print("\nPrediction for adversarial image:")
                    print(f"Predicted digit: {adv_pred}")
                    print("\nTop 3 probabilities:")
                    top3_adv = np.argsort(adv_probs)[-3:][::-1]
                    for digit, prob in zip(top3_adv, adv_probs[top3_adv]):
                        print(f"Digit {digit}: {prob * 100:.2f}%")
                    
                    # Add success/failure message
                    if adv_pred == target_label:
                        print(f"\nSuccess! Model was fooled: {pred} → {target_label}")
                        distortion = tf.reduce_sum(abs(processed_image - adversarial_image)) / float(784)
                        print("Input feature distortion: ", round(float(distortion.numpy()*100), 2), "%")
                    else:
                        print("\nThe model wasn't fooled. Try increasing epsilon or top N.")

                    # Plot both images
                    plot_uploaded_comparison(processed_image, adversarial_image)

                except Exception as e:
                    print(f"Error processing image: {str(e)}")

                new_root.destroy()

        def draw_digit(target_label, epsilon):
            try:
                processed_image = get_drawn_digit()
                print(processed_image)

                # Get prediction for original image
                pred, probs = predict_sample(model, processed_image)

                print("\nPrediction for input image:")
                print(f"Predicted digit: {pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(probs)[-3:][::-1]
                for digit, prob in zip(top3, probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Saliency Map
                adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, processed_image, pred,
                                                                        target_label, epsilon, 0)

                # Get prediction for adversarial image
                adv_pred, adv_probs = predict_sample(model, adversarial_image)

                print("\nPrediction for adversarial image:")
                print(f"Predicted digit: {adv_pred}")
                print("\nTop 3 probabilities:")
                top3_adv = np.argsort(adv_probs)[-3:][::-1]
                for digit, prob in zip(top3_adv, adv_probs[top3_adv]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Add success/failure message
                    if adv_pred == target_label:
                        print(f"\nSuccess! Model was fooled: {pred} → {target_label}")
                        distortion = tf.reduce_sum(abs(processed_image - adversarial_image)) / float(784)
                        print("Input feature distortion: ", round(float(distortion.numpy()*100), 2), "%")
                    else:
                        print("\nThe model wasn't fooled. Try increasing epsilon or top N.")

                # Plot both images
                plot_uploaded_comparison(processed_image, adversarial_image)

            except Exception as e:
                print(f"Error processing image: {str(e)}")

            new_root.destroy()

        def picture(target_label, epsilon):
            try:
                processed_image = preprocess_uploaded_image('/home/designteam10/Pictures/image.jpg')
                print(processed_image)

                # Get prediction for original image
                pred, probs = predict_sample(model, processed_image)

                print("\nPrediction for input image:")
                print(f"Predicted digit: {pred}")
                print("\nTop 3 probabilities:")
                top3 = np.argsort(probs)[-3:][::-1]
                for digit, prob in zip(top3, probs[top3]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Saliency Map
                adversarial_image, num_pixels_changed = create_adversarial_example_saliency(model, processed_image, pred,
                                                                        target_label, epsilon, 0)

                # Get prediction for adversarial image
                adv_pred, adv_probs = predict_sample(model, adversarial_image)

                print("\nPrediction for adversarial image:")
                print(f"Predicted digit: {adv_pred}")
                print("\nTop 3 probabilities:")
                top3_adv = np.argsort(adv_probs)[-3:][::-1]
                for digit, prob in zip(top3_adv, adv_probs[top3_adv]):
                    print(f"Digit {digit}: {prob * 100:.2f}%")

                # Add success/failure message
                    if adv_pred == target_label:
                        print(f"\nSuccess! Model was fooled: {pred} → {target_label}")
                        distortion = tf.reduce_sum(abs(processed_image - adversarial_image)) / float(784)
                        print("Input feature distortion: ", round(float(distortion.numpy()*100), 2), "%")
                    else:
                        print("\nThe model wasn't fooled. Try increasing epsilon or top N.")

                # Plot both images
                plot_uploaded_comparison(processed_image, adversarial_image)

            except Exception as e:
                print(f"Error processing image: {str(e)}")

            new_root.destroy()

        input_frame1 = ttk.Frame(master=new_root)
        epsilon_label = ttk.Label(input_frame1, text="Epsilon:")
        epsilon_entry = ttk.Entry(input_frame1, width=5)
        epsilon_label.pack(side='left', padx=5)
        epsilon_entry.pack(side='left', padx=10)
        input_frame1.pack(pady=10)

        # input_frame2 = ttk.Frame(master=new_root)
        # topn_label = ttk.Label(input_frame2, text="Top N:")
        # topn_entry = ttk.Entry(input_frame2, width=5)
        # topn_label.pack(side='left', padx=5)
        # topn_entry.pack(side='left', padx=10)
        # input_frame2.pack(pady=15)

        input_frame3 = ttk.Frame(master=new_root)
        target_label = ttk.Label(input_frame3, text="Target Class:")
        target_entry = ttk.Entry(input_frame3, width=5)
        target_label.pack(side='left', padx=5)
        target_entry.pack(side='left', padx=10)
        input_frame3.pack(pady=20)

        # Create upload button
        upload_btn = ttk.Button(new_root, text="Upload Image", command=lambda: upload_image(int(target_entry.get()),
                                                                                            float(epsilon_entry.get())))
        upload_btn.pack(pady=25)

        # Draw digit button
        draw_btn = ttk.Button(new_root, text="Draw Digit", command=lambda: draw_digit(int(target_entry.get()),
                                                                                      float(epsilon_entry.get())))
        draw_btn.pack(pady=30)

        # Draw digit button
        picture_btn = ttk.Button(new_root, text="Take Picture", command=lambda: picture(int(target_entry.get()),
                                                                                      float(epsilon_entry.get())))
        picture_btn.pack(pady=35)

        # Run the application
        new_root.mainloop()


    b1 = ttk.Button(root, text="Test MNIST digits across multiple epsilons", command=test_mnist_iterate)
    b1.pack(padx=20, pady=5)
    b2 = ttk.Button(root, text="Test uploaded image across multiple epsilons", command=test_uploaded_iterate)
    b2.pack(padx=20, pady=5)
    b3 = ttk.Button(root, text="Create an adversarial example for random number from MNIST dataset (specify max epsilon)", command=test_mnist_specify)
    b3.pack(padx=20, pady=5)
    b4 = ttk.Button(root, text="Create an adversarial example for uploaded picture (specify max epsilon)", command=test_uploaded_specify)
    b4.pack(padx=20, pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()