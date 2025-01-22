from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

app = FastAPI()

ratings = pd.read_csv('data/ratings.csv')
books = pd.read_csv('data/books.csv')
top_books = joblib.load('models/top_books.joblib')
f_cols = joblib.load('models/f_cols.joblib')
als_model = joblib.load('models/als_model.joblib')
user_features = joblib.load('models/user_features.joblib')
book_features = joblib.load('models/book_features.joblib')
user_item_matrix = joblib.load('models/user_item_matrix.joblib')


class RecommendationRequest(BaseModel):
    model_type: str  # 'top' или 'hybrid'
    user_id: int


class BookClassifier(nn.Module):
    def __init__(self, num_authors, num_favorite_authors, num_other_features, embedding_dim=10):
        super(BookClassifier, self).__init__()
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.favorite_author_embedding = nn.Embedding(num_favorite_authors, embedding_dim)

        self.fc1 = nn.Linear(embedding_dim * 2 + num_other_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        author_embeds = self.author_embedding(x[:, 25].long())
        favorite_author_embeds = self.favorite_author_embedding(x[:, 3].long())

        combined = torch.cat((author_embeds, favorite_author_embeds, x[:, :3], x[:, 4:25], x[:, 26:]), dim=1)
        x = torch.relu(self.bn1(self.fc1(combined)))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


classifier = BookClassifier(3888, 1147, 124)
classifier.load_state_dict(torch.load('models/classifier.pth'))
classifier.eval()


@app.post("/recommend")
async def recommend(req: RecommendationRequest):
    user_id = req.user_id
    model_type = req.model_type
    user_read_books = ratings[ratings['user_id'] == user_id]['book_id'].tolist()

    if user_id not in user_features['user_id'].unique().tolist():

        raise HTTPException(status_code=404, detail="User not found")



    elif model_type == 'top':
        recs = get_top_recommendations(user_read_books)
        return books[books["book_id"].isin(recs)].title.values.tolist()
    elif model_type == 'hybrid':
        recs = get_hybrid_recommendations(user_id, user_read_books)
        return books[books["book_id"].isin(recs)].title.values.tolist()
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")


def get_top_recommendations(read_books):
    recommended_books = [book for book in top_books if book not in read_books]
    return recommended_books[:10]


def get_hybrid_recommendations(user_id, read_books):
    als_rec = list(als_model.recommend(user_id, user_item_matrix[user_id], N=30)[0])
    filt_recs = [book for book in als_rec if book not in read_books]
    recom = pd.DataFrame({'user_id': user_id, 'book_id': filt_recs})
    recom = recom.merge(user_features, on='user_id', how='left')
    recom = recom.merge(book_features, left_on='book_id', right_on='book_id_x', how='left')
    recom = recom[f_cols]
    recom_tensor = torch.tensor(recom.values, dtype=torch.float32)
    with torch.inference_mode():
        recom_outputs = classifier(recom_tensor).squeeze()

    combined = list(zip(filt_recs, recom_outputs.tolist()))
    combined = sorted(combined, key=lambda x: x[1], reverse=True)
    sorted_filt_recs = [book_id for book_id, _ in combined]

    return sorted_filt_recs[:10]


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

