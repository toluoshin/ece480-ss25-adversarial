import tensorflow as tf
from model import build_dnn  # Import the model architecture

def train_model(x_train, y_train):
    model = build_dnn()  # Create the DNN model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        x_train, y_train,
        epochs=10, batch_size=64, validation_split=0.1, verbose=1
    )
    return model
