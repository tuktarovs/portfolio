import os
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_latest_file(directory):
    files = os.listdir(directory)
    latest_file = max(files, key=lambda x: datetime.strptime(x[7:20], '%Y%m%d_%H%M%S'))
    return os.path.join(directory, latest_file)


def load_matrix():
    raw_data_directory = '/app/matrix'
    latest_file_path = get_latest_file(raw_data_directory)
    return np.load(latest_file_path)

def load_data():
#    raw_data_directory = '/app/data'
    raw_data_directory = '/app/data'
    latest_file_path = get_latest_file(raw_data_directory)
    df = pd.read_csv(latest_file_path)
    return df


matrix = load_matrix()
df = load_data()
df["Title"] = df["Title"].str.lower()
sim_matrix = cosine_similarity(matrix)

app = FastAPI()


class UserInput(BaseModel):
    serials: List[str]


def recommend_movie(user_movies, count=5):
    user_movies = [title.lower() for title in user_movies]

    user_ind = df[df['Title'].isin(user_movies)].index
    if len(user_ind) == 0:
        return {'Сериалы не найдены:('}

    user_vec = matrix[user_ind].mean(axis=0).reshape(1, -1)
    sim_scores = cosine_similarity(user_vec, matrix)[0]
    recommend_ind = sim_scores.argsort()[::-1]
    recommend_ind = [i for i in recommend_ind if i not in user_ind][:count]

    return df.iloc[recommend_ind][['Title', 'Genre', 'Description']].to_dict(orient="records")


@app.post('/recommend')
def get_recs(user_inp: UserInput):
    return recommend_movie(user_inp.serials)