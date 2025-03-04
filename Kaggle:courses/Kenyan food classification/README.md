# Food Classification

Task: Build a model to classify Kenyan national dishes (13 classes). The goal is to achieve an accuracy greater than 70%.

## Data
The dataset consists of 8,174 images across 13 classes of Kenyan food.

### Model
A pre-trained food model was found on Hugging Face. The model leverages the Vision Transformer (ViT) from Google, which uses a base architecture with 16x16 patches and an input image size of 224x224 pixels. The pre-trained model was fine-tuned twice: first, the classifier was trained for 10 epochs, and then the entire model was trained for an additional 5 epochs with a lower learning rate.

### Libraries Used
Numpy, Matplotlib, Pandas, Scikit-learn, PIL, OpenCV (cv2), Albumentations, PyTorch, TorchMetrics, Transformers, Datasets

### Model Metrics on Test Data
- Accuracy on validation data: 79.8%
- Accuracy in Kaggle competition: 78%