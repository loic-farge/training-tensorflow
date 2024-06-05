import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Path to the saved model
model_path = 'fruit_classifier_model.keras'

# Load the trained model
model = load_model(model_path)

# Path to the training dataset to get class names
dataset_path = 'fruits-360-original-size/Training'
class_names = sorted(os.listdir(dataset_path))

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(100, 100))  # Load image
    img_array = img_to_array(img) / 255.0  # Convert image to array and normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_fruit(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class]
    
    return predicted_class_name

if __name__ == '__main__':
    # Path to the new image you want to classify
    image_path = 'pear.jpeg'
    
    # Predict the fruit
    predicted_fruit = predict_fruit(image_path)
    
    # Print the prediction
    print(f"The predicted fruit is: {predicted_fruit}")
