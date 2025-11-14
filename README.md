# Multilayer Perceptron (MLP) for MNIST Digit Classification

## Project Overview
This project implements a Multi-Layer Perceptron (MLP), a feedforward artificial neural network, to classify handwritten digits from the MNIST dataset. The goal is to build and train MLP models of varying depths and analyze their performance using metrics such as accuracy, precision, recall, and F1-score.

## Dataset
- MNIST dataset of handwritten digits (0-9)
- 60,000 training images, 10,000 test images
- Images are 28x28 pixels grayscale, normalized to [0,1]

## Features
- Data loading and visualization of samples and pixel intensity
- Implementation of four MLP architectures varying from 1 to 4 hidden layers
- ReLU activations and dropout (rate 0.2) after each hidden layer
- Softmax activation for output layer with 10 neurons (one per class)
- Cross-entropy loss function
- Stochastic Gradient Descent optimizer with learning rate 0.01
- Training for 50 epochs with batch size 50, validation during training
- Saving best model based on validation loss
- Evaluation on test data with metrics: accuracy, precision, recall, F1-score
- Visualization of training curves and confusion matrices for model analysis

## Model Architectures
| Model  | Hidden Layers                |
|--------|-----------------------------|
| Model 1| 1 hidden layer: 512 neurons  |
| Model 2| 2 hidden layers: 512, 256    |
| Model 3| 3 hidden layers: 512, 256, 128|
| Model 4| 4 hidden layers: 512, 256, 128, 64 |

Input layer has 784 neurons (28x28 flattened pixels), and output layer has 10 neurons for digit classes.

## Libraries & Tools
- NumPy: Core numerical computations and neural network implementation
- Matplotlib: Visualizations including sample images, loss curves, and confusion matrices
- PyTorch / torchvision (optional): Used only for loading the MNIST dataset and visualization

## Results
- Best performance achieved by Model 3 (3 hidden layers) with accuracy approximately [specify your accuracy]
- Models show performance gains up to 3 hidden layers; further depth shows diminishing returns
- Common digit misclassifications analyzed via confusion matrix visualization
- Dropout helps reduce overfitting and improves generalization

## Usage
1. Clone the repository
2. Install required Python libraries: `numpy`, `matplotlib`, `torch` (optional)
3. Run the training script to train and validate models
4. Evaluate models on test data and visualize results


