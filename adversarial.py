import tensorflow as tf
import numpy as np
from evaluate_model import predict_sample
import math
import time
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 

# def load_attack_window(root):
#     for widget in root.winfo_children():
#         widget.destroy()
#     fig = Figure(figsize = (10,5), dpi = 100)
#     canvas = FigureCanvasTkAgg()

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

def compute_jacobian(model, adversarial_image, target_label):
    with tf.GradientTape() as tape:
        tape.watch(adversarial_image)
        prediction = model(adversarial_image)
    return tape.jacobian(prediction, adversarial_image)

@tf.function
def update_adversarial_image(saliency_map, adversarial_image, most_salient_pixels, epsilon):
    #indices = [[0, most_salient_pixel]]
    # updates = tf.cast(epsilon, tf.float32) * tf.sign(tf.cast(most_salient_pixels, tf.float32))
    #indices = tf.cast(tf.expand_dims(most_salient_pixels, axis=0), tf.int32)
    #updates = tf.ones_like(most_salient_pixels, dtype=adversarial_image.dtype) * tf.cast(1*epsilon, tf.float32)
    updates = tf.cast(tf.sign(tf.gather(saliency_map, most_salient_pixels)), tf.float32) * tf.cast(1*epsilon, tf.float32)
    batch_indices = tf.zeros_like(most_salient_pixels)
    indices = tf.cast(tf.stack([batch_indices, most_salient_pixels], axis=-1), tf.int32)#tf.cast(tf.expand_dims(most_salient_pixels, axis=0), tf.int32)
    #tf.print("updates: ", updates)
    #tf.print("indices: ", indices)
    return tf.tensor_scatter_nd_add(adversarial_image, indices, updates)
    #return tf.tensor_scatter_nd_add(adversarial_image, tf.cast(tf.expand_dims(most_salient_pixels, axis=1), tf.int32), updates)

def create_adversarial_example_gradual(root, model, input_image, input_label, target_label, epsilon=0.1):
    # clear root
    for widget in root.winfo_children():
        widget.destroy()
    
    # to run GUI event loop
    #plt.ion()
    
    # create subplots
    #fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig = Figure(figsize=(1,1))
    axes = fig.subplots(1,2)

    # Creating the Tkinter canvas containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master = root)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill='both')

    # setting title
    #plt.title("Live Adversarial Attack", fontsize=20)
    axes[0].set_title("Adversarial Image")
    axes[0].axis('off')
    im = axes[0].imshow(input_image.reshape(28, 28), cmap='gray')

    axes[1].set_title("Confidence Scores")
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    _, adv_probs = predict_sample(model, input_image)
    bars = axes[1].bar(digits, adv_probs)
    # axes[1].xlabel("Digits")
    # axes[1].ylabel("Probability")

    #plt.show()

    # variable definitions
    adversarial_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    modified_pixels = set()
    ctr = 0

    # adversarial crafting loop
    while predict_sample(model, adversarial_image)[0] != target_label and ctr < input_image.shape[1]/2:
        ctr+=1

        # compute Jacobian()
        jacobian = compute_jacobian(model, adversarial_image, target_label)

        # generate saliency map
        saliency_map = tf.zeros(adversarial_image.shape[1])

        for i in range(len(saliency_map)):
            target_partial_derivative = jacobian[0][target_label][0][i]
            other_digits_partial_derivative = tf.reduce_sum([jacobian[0, digit, 0, i] for digit in range(10) if digit != target_label])

            if i not in modified_pixels and target_partial_derivative > 0 and other_digits_partial_derivative < 0:
                saliency_value = abs(target_partial_derivative * other_digits_partial_derivative)
                saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value])
            elif i not in modified_pixels and target_partial_derivative < 0 and other_digits_partial_derivative > 0:
                saliency_value = target_partial_derivative * other_digits_partial_derivative
                saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value])
            
        #Create a mask so only digit pixels are affected
        mask = (input_image > 0.05)  # Create mask near non-zero pixels
        mask = tf.cast(tf.reshape(mask, [-1]), dtype=saliency_map.dtype)  # Ensure dtype compatibility
        saliency_map = saliency_map * mask

        # get top 2 salient pixels
        _, top_2_indices = tf.math.top_k(tf.abs(saliency_map), k=2)

        if top_2_indices[0] == 0:       # this basically means that there are no more salient pixels to choose from, aka the model was unable to make an adversarial example
            return adversarial_image, 0

        for pix in top_2_indices:
            modified_pixels.add(pix.numpy().item())
            
        print("number of modified_pixels: ", len(modified_pixels))

        adversarial_image = update_adversarial_image(saliency_map, adversarial_image, top_2_indices, epsilon)
        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

        # Plot adversarial image
        adversarial_display = adversarial_image.numpy().reshape(28, 28)
        im.set_data(adversarial_display)
        

        # Plot confidences
        _, new_probs = predict_sample(model, adversarial_image)
        for bar, new_prob in zip(bars, new_probs):
            bar.set_height(new_prob)

        fig.canvas.draw()
        fig.canvas.flush_events()
        root.update_idletasks()
        #plt.pause(0.01
        
    #plt.ioff()
    # plt.show()
    #plt.close(fig)
    return adversarial_image, len(modified_pixels)

def create_adversarial_example_burst(model, input_image, input_label, target_label, epsilon=0.1, top_n=5):
     # variable definitions
    adversarial_image = tf.convert_to_tensor(input_image, dtype=tf.float32)

    # compute Jacobian()
    jacobian = compute_jacobian(model, adversarial_image, target_label)

    # generate saliency map
    saliency_map = tf.zeros(adversarial_image.shape[1])
    for i in range(len(saliency_map)):
        target_partial_derivative = jacobian[0][target_label][0][i]
        other_digits_partial_derivative = tf.reduce_sum([jacobian[0, digit, 0, i] for digit in range(10) if digit != target_label])

        if target_partial_derivative > 0 and other_digits_partial_derivative < 0:
                saliency_value = abs(target_partial_derivative * other_digits_partial_derivative)
                saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value])
        elif target_partial_derivative < 0 and other_digits_partial_derivative > 0:
            saliency_value = target_partial_derivative * other_digits_partial_derivative
            saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value]) 
    end = time.time()

    #Create a mask around the non-border white pixels
    mask = (input_image > 0.05)  # Create mask near non-zero pixels
    mask = tf.cast(tf.reshape(mask, [-1]), dtype=saliency_map.dtype)  # Ensure dtype compatibility
    saliency_map = saliency_map * mask

    # get top N most salient pixels
    _, top_n_indices = tf.math.top_k(tf.abs(saliency_map), k=top_n)
    
    adversarial_image = update_adversarial_image(saliency_map, adversarial_image, top_n_indices, epsilon)
    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    return adversarial_image, top_n


def create_adversarial_example_saliency(root, model, input_image, input_label, target_label, epsilon=0.1, top_n=5):
    #return create_adversarial_example_burst(model, input_image, input_label, target_label, epsilon, top_n)

    return create_adversarial_example_gradual(root, model, input_image, input_label, target_label, epsilon)
    # input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    # input_label = tf.convert_to_tensor([input_label], dtype=tf.int64)


    # # try to drive to desired target class
    # output_label = tf.convert_to_tensor([target_label], dtype=tf.int64)
    # # print(input_label)
    # # print(output_label)

    # with tf.GradientTape() as tape:
    #     tape.watch(input_image)
    #     prediction = model(input_image)
    #     #print(prediction)
    #     #loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)
    #     loss = -1*tf.keras.losses.sparse_categorical_crossentropy(output_label, prediction) -0.5*tf.keras.losses.sparse_categorical_crossentropy(tf.convert_to_tensor([7]), prediction)

    # #print(loss)
    # #print("input_label:", input_label)

    # # Calculate gradients
    # gradient = tape.gradient(loss, input_image)
    # tf.print("gradient: ", gradient)
    # #print(gradient)

    # # Flatten the gradient to easily identify the top N pixels near the number
    # flat_gradients = tf.reshape(gradient, [-1])

    # # Create a mask around the non-border white pixels
    # mask = (input_image > 0.5)  # Create mask near non-zero pixels
    # mask = tf.cast(tf.reshape(mask, [-1]), dtype=flat_gradients.dtype)  # Ensure dtype compatibility

    # # Apply mask to gradients and get the top N values
    # masked_gradients = flat_gradients * mask
    # _, top_n_indices = tf.math.top_k(tf.abs(masked_gradients), k=top_n)

    # # Generate an empty perturbation array of the same shape as the input
    # perturbation = tf.zeros_like(flat_gradients)

    # # Gather and apply the top N gradients, scaled by epsilon
    # selected_gradients = tf.gather(flat_gradients, top_n_indices)
    # updates = epsilon * tf.sign(selected_gradients)

    # # Update only the top N locations with the perturbations
    # perturbation = tf.tensor_scatter_nd_update(perturbation, tf.expand_dims(top_n_indices, axis=1), updates)

    # # Reshape the perturbation to match the input image's shape and create the adversarial example
    # perturbation = tf.reshape(perturbation, input_image.shape)
    # adversarial_example = input_image + perturbation
    # adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)

    # return adversarial_example

    # NEW CODE
    # convert to TensorFlow
    # input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    # input_image = tf.Variable(input_image, dtype=tf.float32)
    # input_label = tf.convert_to_tensor([input_label], dtype=tf.int64)
    # target_label = tf.convert_to_tensor([target_label], dtype=tf.int64)

    #=================================================================================
#     # variable definitions
    input_tf = tf.convert_to_tensor(input_image, dtype=tf.float32)
    adversarial_image = tf.identity(input_tf)
    square_length = int(math.sqrt(input_image.shape[1]))
    pertubation = tf.zeros_like(adversarial_image)
    max_distortion = top_n * epsilon

   # print("target_label: ", target_label)
    # compute Jacobian()
    start = time.time()
    jacobian = compute_jacobian(model, adversarial_image, target_label)
    end = time.time()
    #print("Jacobian calculation time is :", (end-start), "s")

    # generate saliency map
    saliency_map = tf.zeros(adversarial_image.shape[1])
    start = time.time()
    for i in range(len(saliency_map)):
        target_partial_derivative = jacobian[0][target_label][0][i]
        #print(target_partial_derivative)
        other_digits_partial_derivative = tf.reduce_sum([jacobian[0, digit, 0, i] for digit in range(10) if digit != target_label])
        #other_digits_partial_derivative = sum(jacobian[0][digit][0][i] for digit in range(10) if digit != target_label)
        #print(other_digits_partial_derivative)

        # if target_partial_derivative < 0 or other_digits_partial_derivative > 0:
        #     #saliency_map[i] = 0
        #     saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [0])
        if target_partial_derivative > 0 and other_digits_partial_derivative < 0:
                #saliency_map[i] = target_partial_derivative * abs(other_digits_partial_derivative)
                saliency_value = abs(target_partial_derivative * other_digits_partial_derivative)
                saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value])
        elif target_partial_derivative < 0 and other_digits_partial_derivative > 0:
            #saliency_map[i] = target_partial_derivative * abs(other_digits_partial_derivative)
            saliency_value = target_partial_derivative * other_digits_partial_derivative
            saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value]) 
    end = time.time()
    #print("Saliency generation time is :", (end-start), "s")

    #Create a mask around the non-border white pixels
    mask = (input_image > 0.05)  # Create mask near non-zero pixels
    mask = tf.cast(tf.reshape(mask, [-1]), dtype=saliency_map.dtype)  # Ensure dtype compatibility
    saliency_map = saliency_map * mask

    # make pertubations to top N
    _, top_n_indices = tf.math.top_k(tf.abs(saliency_map), k=top_n)
    #tf.print("top_n_indices: ", top_n_indices)
    #selected_pixels = tf.gather(saliency_map, top_n_indices)
    #tf.print("selected_pixels: ", selected_pixels)
    

    #pertubation = tf.tensor_scatter_nd_add(pertubation, [[0, most_salient_pixel]], [epsilon])
    # perturbation = tf.tensor_scatter_nd_update(perturbation, tf.expand_dims(top_n_indices, axis=1), updates)
    adversarial_image = update_adversarial_image(saliency_map, adversarial_image, top_n_indices, epsilon)

    adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

    return adversarial_image
#     #=================================================================================

    # variable definitions
    input_tf = tf.convert_to_tensor(input_image, dtype=tf.float32)
    adversarial_image = tf.identity(input_tf)
    square_length = int(math.sqrt(input_image.shape[1]))
    pertubation = tf.zeros_like(adversarial_image)
    max_distortion = top_n * epsilon

    modified_pixels = set()
    ctr = 0

    # adversarial crafting loop
    while predict_sample(model, adversarial_image)[0] != target_label and ctr < top_n/2:#tf.reduce_sum(pertubation) <= max_distortion:  #sum(perturbation[i] for i in range(len(perturbation))) <= max_distortion:
        ctr+=1
        # # compute gradient of each output class
        # with tf.GradientTape() as tape:
        #     tape.watch(adversarial_image)
        #     prediction = model(adversarial_image)

        # # calculate the Jacobian matrix
        # start = time.time()
        # jacobian = tape.jacobian(prediction, adversarial_image)
        # end = time.time()
        # print("Jacobian calculation time is :", (end-start), "s")
        # #print(jacobian[0][target_label][0])
        # # print(jacobian.shape)

        # compute Jacobian()
        start = time.time()
        jacobian = compute_jacobian(model, adversarial_image, target_label)
        end = time.time()
        print("Jacobian calculation time is :", (end-start), "s")

        # generate saliency map
        #saliency_map = np.zeros(adversarial_image.shape[1])
        saliency_map = tf.zeros(adversarial_image.shape[1])
        most_salient_pixel = 0

        start = time.time()
        for i in range(len(saliency_map)):
            target_partial_derivative = jacobian[0][target_label][0][i]
            #print(target_partial_derivative)
            other_digits_partial_derivative = tf.reduce_sum([jacobian[0, digit, 0, i] for digit in range(10) if digit != target_label])
            #other_digits_partial_derivative = sum(jacobian[0][digit][0][i] for digit in range(10) if digit != target_label)
            #print(other_digits_partial_derivative)

            # if i in modified_pixels or target_partial_derivative < 0 or other_digits_partial_derivative > 0:
            #     #saliency_map[i] = 0
            #     saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [0])
            if i not in modified_pixels and target_partial_derivative > 0 and other_digits_partial_derivative < 0:
                #saliency_map[i] = target_partial_derivative * abs(other_digits_partial_derivative)
                saliency_value = abs(target_partial_derivative * other_digits_partial_derivative)
                saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value])
            elif i not in modified_pixels and target_partial_derivative < 0 and other_digits_partial_derivative > 0:
                #saliency_map[i] = target_partial_derivative * abs(other_digits_partial_derivative)
                saliency_value = target_partial_derivative * other_digits_partial_derivative
                saliency_map = tf.tensor_scatter_nd_update(saliency_map, [[i]], [saliency_value])
            
            # if saliency_map[i] > saliency_map[most_salient_pixel]:      # keep track of the most salient pixel
            #     most_salient_pixel = i
        end = time.time()
        print("Loop time is :", (end-start), "s")

        #Create a mask around the non-border white pixels
        mask = (input_image > 0.05)  # Create mask near non-zero pixels
        mask = tf.cast(tf.reshape(mask, [-1]), dtype=saliency_map.dtype)  # Ensure dtype compatibility
        saliency_map = saliency_map * mask

        # make pertubations to top N
        _, top_n_indices = tf.math.top_k(tf.abs(saliency_map), k=2)

        for pix in top_n_indices:
            modified_pixels.add(pix.numpy().item())
            tf.print("saliency val: ", saliency_map[pix.numpy().item()])
        tf.print("top_n_indices: ", top_n_indices)
        print("number of modified_pixels: ", len(modified_pixels))

        #selected_pixels = tf.gather(saliency_map, top_n_indices)
        #tf.print("selected_pixels: ", selected_pixels)
        

        #pertubation = tf.tensor_scatter_nd_add(pertubation, [[0, most_salient_pixel]], [epsilon])
        # perturbation = tf.tensor_scatter_nd_update(perturbation, tf.expand_dims(top_n_indices, axis=1), updates)
        adversarial_image = update_adversarial_image(saliency_map, adversarial_image, top_n_indices, epsilon)

        adversarial_image = tf.clip_by_value(adversarial_image, 0, 1)

        #print(saliency_map)
        #print("most salient: ",saliency_map[most_salient_pixel])
        # make pertubation
        #perturbation[most_salient_pixel] += epsilon
        # pertubation = tf.tensor_scatter_nd_add(pertubation, [[0, most_salient_pixel]], [epsilon])
        # adversarial_image = update_adversarial_image(adversarial_image, most_salient_pixel, epsilon)
        #adversarial_image.assign(tf.tensor_scatter_nd_update(adversarial_image, 
         #   [[0, most_salient_pixel]], [epsilon])
        #adversarial_image[(most_salient_pixel)//square_length, most_salient_pixel%square_length].assign_add(epsilon)

            
            
            
            



    return adversarial_image