import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import Counter

dataset_path = "fruits-360-original-size/Training"

# Function to load and preprocess the images
def load_and_preprocess_images(folder, num_images=None):
    images = []
    labels = []
    filenames = []
    for label, subfolder in enumerate(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        count = 0
        for filename in os.listdir(subfolder_path):
            if num_images and count >= num_images:
                break
            img_path = os.path.join(subfolder_path, filename)
            img = load_img(img_path, target_size=(100, 100))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
            filenames.append((subfolder, filename))
            count += 1
    return np.array(images), np.array(labels), filenames

# Load and preprocess the training data
x_train, y_train, train_filenames = load_and_preprocess_images(dataset_path, num_images=500)

# Print some filenames to ensure diversity
print("Training filenames sample:", train_filenames[:10])

# Print count of images per class
print("Training labels distribution:", Counter(y_train))

# Convert labels to categorical format
y_train = to_categorical(y_train, num_classes=len(os.listdir(dataset_path)))

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_val shape: {x_val.shape}, y_val shape: {y_val.shape}")

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(os.listdir(dataset_path)), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Define the number of epochs and batch size
epochs = 20  # Reduced for quicker testing
batch_size = 32

# Train the model
history = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    validation_data=(x_val, y_val))

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Path to the test dataset directory
test_dataset_path = 'fruits-360-original-size/Test'
class_names = sorted(os.listdir(dataset_path))

# Load and preprocess the test data
x_test, y_test, test_filenames = load_and_preprocess_images(test_dataset_path)

# Print some filenames to ensure diversity
print("Test filenames sample:", test_filenames[:10])

# Print count of images per class
print("Test labels distribution:", Counter(y_test))

# Convert labels to categorical format
y_test = to_categorical(y_test, num_classes=len(os.listdir(dataset_path)))

# Print the shapes of the test datasets
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test data
predictions = model.predict(x_test)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Ensure we display one image from each subfolder
unique_labels = set()
unique_indices = []
for i, label in enumerate(true_labels):
    if label not in unique_labels:
        unique_labels.add(label)
        unique_indices.append(i)
    if len(unique_labels) == 10:  # Change the number based on how many you want to display
        break

# Display the selected test images with their predicted and true labels
plt.figure(figsize=(15, 5))
for i, index in enumerate(unique_indices):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[index])
    plt.xlabel(f"Pred: {class_names[predicted_labels[index]]}\nTrue: {class_names[true_labels[index]]}")
plt.show()

# Save the model in native Keras format
model.save('fruit_classifier_model.keras')
