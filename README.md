# Fruit Classifier Project

This project demonstrates how to build a Convolutional Neural Network (CNN) to classify images of various fruits. The project includes scripts for training the model and predicting the fruit type from an image.

## Prerequisites

- Python 3.7 or higher
- Git
- Virtualenv (optional, but recommended)

## Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/loic-farge/training-tensorflow.git
cd fruit-classifier
```

### Step 2: Set Up Virtual Environment (Optional)

Create a virtual environment to manage dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 4: Download the Training Set

Download the Fruits 360 dataset from Kaggle:

1. Go to [Fruits 360 Dataset on Kaggle](https://www.kaggle.com/moltean/fruits).
2. Download the dataset and extract it to a folder named `fruits-360-original-size` in the project root directory.

Your directory structure should look like this:

```
fruit-classifier/
├── fruits-360-original-size/
│   ├── Training/
│   └── Test/
├── main.py
├── predict_fruit.py
├── requirements.txt
└── README.md
```

### Step 5: Train the Model

Use the `main.py` script to train the model:

```bash
python main.py
```

This script will load the training data, preprocess it, define the CNN model, train the model, and save it to a file named `fruit_classifier_model.keras`.

### Step 6: Make Predictions

Use the `predict_fruit.py` script to make predictions on new images:

1. Place the image you want to classify in the project directory.
2. Update the `image_path` variable in `predict_fruit.py` to the path of your image.
3. Run the script:

```bash
python predict_fruit.py
```

This script will load the trained model, preprocess the input image, and print the predicted fruit type.

## Files in the Repository

- `main.py`: Script to train the CNN model using the Fruits 360 dataset.
- `predict_fruit.py`: Script to load the trained model and make predictions on new images.
- `requirements.txt`: List of required Python packages.

## Notes

- Ensure the `fruits-360-original-size` directory is not tracked by Git by adding it to `.gitignore`.
- This project is for training purposes and may not work wonderfully. The dataset and model architecture may need improvements for better performance.
