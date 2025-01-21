import warnings
import dill
import pandas as pd


from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from os import PathLike
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

app = FastAPI()

model_name_like = 'model_*.pkl'


def load_model(folder: PathLike = 'models') -> BaseEstimator:
    folder = Path(folder)
    files = list(folder.glob(model_name_like))

    if files:
        last_file = sorted(files)[-1]
        with open(last_file, 'rb') as file:
            model = dill.load(file)
        return model

    else:
        raise FileNotFoundError('Нет моделей')


model = load_model()


class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    client_id: str
    execution_time: str
    prediction: int


@app.get('/status')
def status():
    return "Hello! I'm ok!!"


@app.get('/version')
def version():
    return model.metadata


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    start_time = datetime.now()
    data = pd.DataFrame.from_dict([form.dict()])
    y = model.predict_proba(data)[:, 1] > model.metadata['metrics']['threshold']
    execution_time = str(datetime.now() - start_time)
    print(int(y[0]))

    return {
        'session_id': form.session_id,
        'client_id': form.client_id,
        'execution_time': execution_time,
        'prediction': int(y[0])
    }
