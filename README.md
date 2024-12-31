import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset
def load_dataset():
    # Replace with the correct file path
    try:
        data = pd.read_csv("path_to_your_dataset.csv")
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: Dataset file not found. Check the file path.")
        exit()
    features = data.iloc[:, :-1].values  # Adjust column indexing if needed
    labels = data.iloc[:, -1].values
    return features, labels

# Preprocess data
def preprocess_data(features, labels):
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    
    return features, labels, label_encoder

# Build model
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True, activation='relu'),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Main program
if _name_ == "_main_":
    # Load and preprocess the data
    features, labels = load_dataset()
    features, labels, label_encoder = preprocess_data(features, labels)
    
    # Reshape features for LSTM input
    features = features.reshape((features.shape[0], 1, features.shape[1]))
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Build and train the model
    num_classes = labels.shape[1]
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, num_classes)
    
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
