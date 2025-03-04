# First Experience in NLP

**Task**: To sort emails into spam or ham based on the content of the text.

**Key Metric**: F1-score > 0.8

## Data
The dataset used for this project is the "Emails for Spam or Ham Classification" from Trec 2006, comprising a collection of emails labeled as spam or legitimate (ham). The dataset contains [укажите общее количество сообщений и их распределение].

## Model
The performance of BERT and GPT-2 was compared to generate embeddings for the classification task. The classifier consists of two linear layers with ReLU activation between them and a sigmoid activation function at the output. Additionally, Batch Normalization was employed to improve convergence, along with Dropout for regularization.

### Metrics on Test Data:
- **BERT**:
  - Accuracy: 97%
  - F1-score: 95.6%
  
- **GPT-2**:
  - Accuracy: 91.5%
  - F1-score: 87%

## Conclusion
This project successfully demonstrated the efficacy of transformer-based models in the email spam classification task. The results indicate that BERT outperformed GPT-2 in terms of both accuracy and F1-score, highlighting the effectiveness of contextual embeddings in text classification tasks.