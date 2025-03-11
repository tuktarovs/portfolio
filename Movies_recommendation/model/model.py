import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def get_latest_file(directory):
    files = os.listdir(directory)
    latest_file = max(files, key=lambda x: datetime.strptime(x[7:21], '%Y%m%d_%H%M%S'))
    return os.path.join(directory, latest_file)


def load_data():
#    raw_data_directory = '/app/data'
    raw_data_directory = '/app/data'
    latest_file_path = get_latest_file(raw_data_directory)
    df = pd.read_csv(latest_file_path)
    return df


def matrix():
    def preprocess_text(text):
        if not isinstance(text, str):
            return ''
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    def combine(row):
        description = preprocess_text(row['Description'])
        genre = preprocess_text(row['Genre'])
        keywords = preprocess_text(row['Keywords'])
        actors = " ".join([f'{preprocess_text(actor)}' for actor in row['Actors']])
        return f'{description} {keywords} {actors}'

    df = load_data()
    df['Title'] = df['Title'].apply(preprocess_text)
    df['Genre'] = df["Genre"].apply(preprocess_text)
    df['Keywords'] = df['Keywords'].str.replace(',', ' ', regex=False)
    df['features'] = df.apply(combine, axis=1)

    year_matrix = MinMaxScaler().fit_transform(
        df["Release Year"].fillna(df["Release Year"].median()).values.reshape(-1, 1)    )

    def genre_matrix(df):
        df["Genre"] = df["Genre"].apply(lambda x: x.split(" ") if isinstance(x, str) else [])
        mlb = MultiLabelBinarizer()
        return mlb.fit_transform(df["Genre"])

    genre_matrix = genre_matrix(df)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    bert_matrix = model.encode(df['features'].tolist(), convert_to_numpy=True)

    final_matrix = np.hstack([bert_matrix, genre_matrix, year_matrix])
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join('/app/matrix', f'matrix_{current_time}.npy')
    np.save(file_path, final_matrix)


if __name__ == '__main__':
    matrix()


