import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import os
from datetime import datetime
import numpy as np
from models.preprocessing import missings, new_features


app = FastAPI()


def get_latest_file(directory):

    files = os.listdir(directory)
    latest_file = max(files, key=lambda x: datetime.strptime(x[6:20], '%Y%m%d_%H%M%S'))
    return os.path.join(directory, latest_file)


def load_model():
    raw_data_directory = '/app/models/pipe'
    latest_file_path = get_latest_file(raw_data_directory)
    return load(latest_file_path)


class Geo(BaseModel):
    lat: float
    lng: float


class input_data(BaseModel):
    id: int | None = None
    floor: int
    rooms_count: int
    total_area: float
    kitchen_area: float
    year: int
    parking: str
    total_floors: int
    lat: float
    lng: float
    balconiesCount: int | None = 0
    materialType: str


@app.get("/")
def home():

    return {"message": "API для оценки стоимости квартиры"}


@app.post("/predict/")
def predict(data: input_data):
    model = load_model()

    if model is None:
        return {'Модель не найдена'}
    else:
        data = pd.DataFrame([data.dict()])
#        data['distance_from_center'] = data.apply(calculate_distance, axis=1)
#        data['floor_status'] = data.apply(determine_floor_status, axis=1).astype('category')
        prediction = model.predict(data)
        return {'прогнозируемая цена': prediction.tolist()}
