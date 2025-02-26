import tensorflow as tf
import numpy as np

def create_adversarial_example(model, input_image, input_label, epsilon=0.1):
    input_image = tf.convert_to_tensor(input_image)
    input_label = tf.convert_to_tensor(input_label)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

    #print(input_label)
    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    adversarial_example = input_image + epsilon * signed_grad
    adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)

    return adversarial_example

def create_adversarial_example_saliency(model, input_image, input_label, target_label, epsilon=0.1, top_n=5):
    input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    input_label = tf.convert_to_tensor(input_label, dtype=tf.int64)


    # try to drive to desired target class
    output_label = tf.convert_to_tensor(target_label, dtype=tf.int64)
    # print(input_label)
    # print(output_label)

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        #loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)
        loss = -tf.keras.losses.sparse_categorical_crossentropy(output_label, prediction)

    #print("input_label:", input_label)

    # Calculate gradients
    gradient = tape.gradient(loss, input_image)

    # Flatten the gradient to easily identify the top N pixels near the number
    flat_gradients = tf.reshape(gradient, [-1])

    # Create a mask around the non-border white pixels
    mask = (input_image > 0.5)  # Create mask near non-zero pixels
    mask = tf.cast(tf.reshape(mask, [-1]), dtype=flat_gradients.dtype)  # Ensure dtype compatibility

    # Apply mask to gradients and get the top N values
    masked_gradients = flat_gradients * mask
    _, top_n_indices = tf.math.top_k(tf.abs(masked_gradients), k=top_n)

    # Generate an empty perturbation array of the same shape as the input
    perturbation = tf.zeros_like(flat_gradients)

    # Gather and apply the top N gradients, scaled by epsilon
    selected_gradients = tf.gather(flat_gradients, top_n_indices)
    updates = epsilon * tf.sign(selected_gradients)

    # Update only the top N locations with the perturbations
    perturbation = tf.tensor_scatter_nd_update(perturbation, tf.expand_dims(top_n_indices, axis=1), updates)

    # Reshape the perturbation to match the input image's shape and create the adversarial example
    perturbation = tf.reshape(perturbation, input_image.shape)
    adversarial_example = input_image + perturbation
    adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)

    return adversarial_example