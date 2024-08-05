# MNIST Handwritten Digit Classification
This repository contains code for training a neural network to classify handwritten digits from the MNIST dataset. The code leverages TensorFlow and Keras to build, train, and evaluate the model. Additionally, it includes visualization of the training process and predictions on test samples.

# Features

## Data Preprocessing
### Dataset Splitting:
The MNIST dataset is split into training, validation, and test sets to evaluate the model's performance accurately.
### Normalization: 
Input images are normalized to the [0, 1] range to facilitate faster and more stable training.

## Model Architecture
### Sequential Model:
A sequential neural network with three dense layers, including dropout and batch normalization for regularization.

### Layer Details:
Flatten: Converts 28x28 images into 1D vectors.
Dense(64, activation='relu'): First dense layer with 64 units and ReLU activation.
BatchNormalization and Dropout(0.2): Applied after the first and second dense layers.
Dense(32, activation='relu'): Second dense layer with 32 units and ReLU activation.
Dense(10, activation='linear'): Output layer with 10 units corresponding to the digit classes.

## Training and Evaluation
### Callbacks: 
Early stopping and learning rate reduction to prevent overfitting and improve model performance.
### Visualization: 
Plots showing training and validation loss over epochs.
### Evaluation Metrics: 
Test loss and accuracy are reported after training.

## Sample Prediction
### Visualization: 
Displays sample test images along with their true and predicted labels.
### Misclassification Count: 
Counts and displays the number of misclassified samples.

## Dependencies
TensorFlow: For building and training the neural network.
NumPy: For numerical operations and array manipulation.
Matplotlib: For visualization of the training process and predictions.
scikit-learn: For splitting the dataset into training and validation sets.

## Results:
![image](https://github.com/user-attachments/assets/54a080fc-6bdd-4bff-a252-c963b526bb15)
![image](https://github.com/user-attachments/assets/29360ba8-150c-4ec7-a596-f1db59e8e72f)
![image](https://github.com/user-attachments/assets/d27ab293-3526-4a70-97a5-b5a443a20aac)



