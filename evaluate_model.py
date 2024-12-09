import numpy as np

def evaluate_model(model, x_test, y_test):
    # Evaluate the trained model
    print("\nEvaluating the model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

def predict_sample(model, sample_image):
    # Predict on a single sample
    prediction = model.predict(sample_image, verbose=0)
    return np.argmax(prediction), prediction[0]
