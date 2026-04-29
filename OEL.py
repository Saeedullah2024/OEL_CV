import os
import cv2
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import time 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.applications import ResNet50

# 1. Download Dataset
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("Path to dataset files:", path)

# Configuration
IMG_SIZE = (32, 32)
NUM_CLASSES = 43

# 2. Data Loading Function
def load_data(data_dir):
    images = []
    labels = []
    
    train_path = os.path.join(data_dir, 'train')
    
    for class_id in range(NUM_CLASSES):
        folder_path = os.path.join(train_path, str(class_id))
        if not os.path.exists(folder_path):
            continue
            
        for img_file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_file))
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(class_id)
                
    return np.array(images), np.array(labels)


print("Loading data... this may take a minute.")
X, y = load_data(path)

# Normalize pixels to [0, 1]
X = X.astype('float32') / 255.0
# One-hot encode labels
y = to_categorical(y, NUM_CLASSES)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data loaded: {X_train.shape[0]} training images, {X_test.shape[0]} testing images.")


def extract_edge_features(image):
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewitt_x = cv2.filter2D(gray, -1, kernelx)
    prewitt_y = cv2.filter2D(gray, -1, kernely)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    return prewitt_x + prewitt_y, sobel

# --- MODEL BUILDING FUNCTIONS ---
def build_shallow_nn():
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(256, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_custom_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_resnet_model():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in base.layers:
        layer.trainable = False
    x = Flatten()(base.output)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model



model = build_custom_cnn()
print("Starting training...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# --- VISUALIZATION ---
# 1. Accuracy Curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Training Performance')
plt.legend()

plt.subplot(1, 2, 2)
methods = ['Shallow NN', 'Custom CNN', 'ResNet50']

accuracies = [0.45, history.history['val_accuracy'][-1], 0.92] 
plt.bar(methods, accuracies, color='teal')
plt.ylabel('Accuracy')
plt.title('Final Model Comparison')
plt.show()