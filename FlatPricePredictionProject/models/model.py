import os
import pandas as pd
from joblib import dump
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from datetime import datetime
from preprocessing import missings, new_features
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.outliers import Winsorizer
from feature_engine.selection import DropFeatures
from feature_engine.encoding import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import optuna
from catboost import CatBoostRegressor



def get_latest_file(directory):
    files = os.listdir(directory)
    latest_file = max(files, key=lambda x: datetime.strptime(x[7:21], '%Y%m%d_%H%M%S'))
    return os.path.join(directory, latest_file)


def load_data():
    raw_data_directory = '/app/data/raw'
    latest_file_path = get_latest_file(raw_data_directory)
    df = pd.read_csv(latest_file_path)
    return df


def hyperparameters(X_train,y_train):
    def objective(trial):
        params = {
            'iterations': 1000,
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
            'depth': trial.suggest_int('depth', 1, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_seed': 42,
            'eval_metric': 'MAPE',
            'verbose': 0
        }

        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        mape_scores = []

        for train_index, val_index in kf.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            model = CatBoostRegressor(**params)
            model.fit(X_train_fold, y_train_fold, eval_set=(X_val_fold, y_val_fold), verbose=0)

            y_val_pred = model.predict(X_val_fold)

            mape = mean_absolute_percentage_error(y_val_fold, y_val_pred)
            mape_scores.append(mape)
        return np.mean(mape_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    return study.best_params

def train_model():
    df = load_data()
    df = df.drop_duplicates()
    X = df.drop(columns=['price'])
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = Pipeline(steps=[
        ('imputer', FunctionTransformer(missings)),
        ('engineer', FunctionTransformer(new_features)),
        ('dropper', DropFeatures(['id', 'lat', 'lng'])),
        ('outliers', Winsorizer()),
        ('normalization', YeoJohnsonTransformer()),
        ('oneHotEncoder', OneHotEncoder(drop_last_binary=True))]
    )
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    best_params = hyperparameters(X_train_preprocessed, y_train)
    cb_model = CatBoostRegressor(**best_params)
    cb_model.fit(X_train_preprocessed, y_train)
    test_prediction = cb_model.predict(X_test_preprocessed)
    mae = mean_absolute_error(y_test, test_prediction)
    mape = mean_absolute_percentage_error(y_test, test_prediction)


    pipe = Pipeline(steps=[
        ('imputer', FunctionTransformer(missings)),
        ('engineer', FunctionTransformer(new_features)),
        ('dropper', DropFeatures(['id', 'lat', 'lng'])),
        ('outliers', Winsorizer()),
        ('normalization', YeoJohnsonTransformer()),
        ('oneHotEncoder', OneHotEncoder(drop_last_binary=True)),
        ('model', CatBoostRegressor(**best_params))])
    pipe.fit(X, y)
    pipe = Pipeline(steps=[step for step in pipe.steps if step[0] not in ['imputer']])
    pipe.metadata = {
        'name': 'Flat price predictor',
        'description': 'Flat_price_prediction',
        'model': pipe['model'].__class__.__name__,
        'version': 1.0,
        'model_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'author': 'Sergey Tuktarov',
        'metrics': {
            'mae': mae,
            'mse': mape}
    }

    model_dir = '/app/models/pipe'
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dump(pipe, os.path.join(model_dir, f'model_{current_time}.joblib'))

    print("Модель и метаданные успешно сохранены.")


if __name__ == '__main__':
    train_model()