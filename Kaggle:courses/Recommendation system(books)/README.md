# Recommendation System

## Objective
To provide meaningful book recommendations to users based on their previous rating history.

## Dataset
Utilized the Goodreads Books dataset, which contains approximately 6 million ratings for 10,000 books.

## Libraries
- Pandas
- NumPy
- SciPy
- Implicit
- Scikit-learn
- PyTorch
- Joblib
- FastAPI
- Pydantic

## Models
Two models were developed:
1. Popularity-Based Model: Recommends 10 books with the highest number of positive ratings that the user has not yet read.
2. Hybrid Model: Recommends 10 books based on the user's previous preferences. The first part of the model employs collaborative filtering, while the second part utilizes a content-based approach.

Following the training phase, A/B testing was conducted. The results indicated that the popularity-based model significantly outperformed the hybrid model. It is important to note that popular books inherently had a higher chance of being included in the test sample.

## Application
As a result of this project, a web application was created that accepts input for the desired recommendation model ('top' for the popularity-based model or 'hybrid' for the hybrid model) and the user ID. The application outputs a list of 10 recommended books based on the selected model.

