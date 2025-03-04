# Handwritten Character Recognition (EMNIST)

**Task**: To classify EMNIST 'balanced' characters using Convolutional Neural Networks (CNN). The model is designed to identify characters drawn through an application.

## Data
The dataset comprises 47 black-and-white symbols representing the English alphabet and digits. For each of the 47 symbols, there are 2400 examples in the training set and 400 in the test set, totaling 131,600 images.

## Model Architecture
The final model features:
- Two convolutional layers with ReLU activation functions followed by batch normalization.
- Dimensionality reduction via Average Pooling.
- A classifier with three sequential linear layers, utilizing ReLU activation for the interstitial layers, with dropout applied at a rate of 0.5 for regularization.

### Hyperparameters of the Best Model:
- Convolutional Layer Kernel: (3, 3)
- Stride: 1
- Classifier:
  - Linear Layer 1: 512 neurons
  - Linear Layer 2: 256 neurons
  - Linear Layer 3: Classifier output with 47 neurons (corresponding to the number of unique symbols)

### Libraries Used
- **Torch**: For implementing the convolutional neural network.
- **Torchvision**: For data loading and preprocessing.
- **Matplotlib**: For visualizing the training process and results.

### Model Metrics on Test Data
The accuracy of the best model on the test set reached 88.8%.

## Installation and Service Launch
To run the service locally, execute the following commands:
```bash
git clone https://github.com/tuktarovs/portfolio/CV_classification.git
cd portfolio/CV_classification
docker build -t symb_cnn .
docker run -p 8000:8000 symb_cnn