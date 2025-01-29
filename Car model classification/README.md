# Car Model Classification

Task: Build two models for classifying cars into 10 classes:
1. A basic model without using transfer learning.
2. A model fine-tuned from a pre-trained network.

The goal is to achieve an F1-score metric of at least 70%.

## Data
The dataset contains 10,000 images of cars of various sizes. The data is organized into folders corresponding to each class.

## Architecture of the Basic Model
The basic model is a convolutional neural network (CNN) developed using the PyTorch library. It includes several convolutional layers to extract features from the input images, as well as max pooling layers to reduce dimensionality. Skip connections are used to retain information and improve training convergence. The model features adaptive global average pooling to simplify the output dimensionality to a fixed size. Fully connected layers perform the final classification. A dropout layer is employed to prevent overfitting. Data augmentation was also applied during the training of the basic model. The basic model was trained for 55 epochs.

### Pre-trained Model
The model leverages the Vision Transformer (ViT) from Google, which uses a base architecture with 16x16 patches and an input image size of 224x224 pixels. The pre-trained model was fine-tuned twice: first, the classifier was trained for 10 epochs, and then the entire model was trained for an additional 7 epochs with a lower learning rate.

### Libraries Used
Numpy, Matplotlib, Pandas, Scikit-learn, PIL, Albumentations, Torch, PyTorch Lightning, Torchvision, Torchmetrics, Transformers.

### Model Metrics on Test Data
- F1-score (macro) of the basic model: 80.5%
- Result of the pre-trained model: 92.4%

### Conclusion
Both models successfully tackled the task of car classification with high accuracy. The results demonstrate the effectiveness of using pre-trained models in image classification tasks, significantly improving the outcomes compared to models trained from scratch.