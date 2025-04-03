import numpy as np
import tensorflow as tf

def evaluate_model(model, x_test, y_test):
    # Evaluate the trained model
    print("\nEvaluating the model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

#def predict_sample(interpreter, sample_image):
    # Predict on a single sample
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()

    # input_data = np.array(sample_image, dtype=np.float32)
    # interpreter.set_tensor(input_details[0]['index'], input_data)
    # interpreter.invoke()

    # output = interpreter.get_tensor(output_details[0]['index'])
    # predicted_class = int(np.argmax(output))
    # return predicted_class, output[0]

def predict_sample(model, sample_image):
    # Predict on a single sample
    prediction = model.predict(sample_image, verbose=0)
    return np.argmax(prediction), prediction[0]