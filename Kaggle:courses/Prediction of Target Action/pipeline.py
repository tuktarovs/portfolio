import warnings
import dill

import pandas as pd

from sklearn.pipeline import Pipeline
from feature_engine.selection import DropDuplicateFeatures, DropCorrelatedFeatures, DropFeatures, DropConstantFeatures
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.outliers import Winsorizer
from feature_engine.wrappers import SklearnTransformerWrapper
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.base import BaseEstimator
from lightgbm import LGBMClassifier

from datetime import datetime
from sklearn.preprocessing import FunctionTransformer, StandardScaler

warnings.filterwarnings('ignore')


def model_creation():
    df_full = get_data()
    print('данные загружены')
    model = get_model()
    print('модель загружена')
    metadata = get_metadata(model, df_full)
    print('метаданные получены', metadata)
    save_model(model, metadata)
    print('модель сохранена')


def get_data() -> pd.DataFrame:
    with open(f'data/ga_hits.pkl', 'rb') as file:
        ga_hits = dill.load(file)
    with open(f'data/ga_sessions.pkl', 'rb') as file:
        ga_sessions = dill.load(file)

    event = ['sub_car_claim_click', 'sub_car_claim_submit_click',
             'sub_open_dialog_click', 'sub_custom_question_submit_click',
             'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success',
             'sub_car_request_submit_click']
    ga_hits['event_value'] = ga_hits.apply(lambda x: 1 if x.event_action in event else 0, axis=1)
    ga_hits = ga_hits[['session_id', 'event_value']]
    hits = ga_hits.groupby(by='session_id').sum()
    hits['event_value'] = hits.apply(lambda x: 1 if x.event_value > 0 else 0, axis=1)
    df_full = pd.merge(left=ga_sessions, right=hits, on='session_id', how='inner')
    return df_full


def get_model() -> BaseEstimator:
    return Pipeline(steps=[
        ('imputer', FunctionTransformer(missings)),
        ('engineer', FunctionTransformer(new_features)),
        ('dropper', DropFeatures(['client_id', 'visit_date', 'visit_time', 'device_screen_resolution'])),
        #    ('normalization', BoxCoxTransformer()),
        ('normalization', YeoJohnsonTransformer()),
        ('outliers', Winsorizer()),
        ('scaler', SklearnTransformerWrapper(StandardScaler())),
        ('rare', RareLabelEncoder(tol=0.024244275099912872)),
        ('oneHotEncoder', OneHotEncoder(drop_last_binary=True)),
        ('bool_convert', FunctionTransformer(convert_float)),
        ('constantDropper', DropConstantFeatures(tol=0.9763040173185428)),
        ('duplicatedDropper', DropDuplicateFeatures()),
        ('correlatedDropper', DropCorrelatedFeatures(threshold=0.9419706591416945)),
        ('model', LGBMClassifier(n_estimators=1464,
                                 num_leaves=28,
                                 learning_rate=0.06441747665717384,
                                 reg_lambda=29.600022434984556,
                                 reg_alpha=26.375317773499894,
                                 min_child_samples=44,
                                 boosting_type='goss',
                                 random_state=42))
    ])


def missings(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    missing_values = [float('nan'), '(none)', '(not set)', '0x0']

    if 'device_screen_resolution' in data.columns:
        data['device_screen_resolution'] = data['device_screen_resolution'].replace(missing_values,
                                                                                    data.device_screen_resolution.mode()[
                                                                                        0])

    if 'geo_country' in data.columns:
        data['geo_country'] = data['geo_country'].replace(missing_values, data.geo_country.mode()[0])

    if 'device_os' in data.columns:
        data.device_os[(data.device_os.isna()) & (data['device_brand'] == 'Apple')] = 'iOS'
        data.device_os[(data.device_os.isna()) & (data['device_brand'] != '') & (data['device_brand'] != '(not set)')] = 'Android'

    return data.fillna('(not set)')


def new_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    import csv
    with open('data/distance_from_moscow.csv', encoding='utf-8') as file:
        distance = csv.reader(file)
        next(distance)
        distance_from_moscow = {line[0]: float(line[1]) for line in distance}

    def distance(city: str) -> float:
        return distance_from_moscow.get(str(city).lower(), -1)

    organic = ['organic', 'referral', '(none)']
    social = ['QxAxdyPLuQMEcrdZWdWb', 'MvfHsxITijuriZxsqZqt', 'ISrKoXQCxqqYvAZICvjs', 'IZEXUFLARCUMynmHNBGo',
              'PlbkrSYoHuZBWfYjYnfw', 'gVRrcxiDQubJiljoTbGm']
    ussr = ['Azerbaijan', 'Armenia', 'Belarus', 'Georgia', 'Kazakhstan', 'Kyrgyzstan', 'Latvia', 'Lithuania', 'Moldova',
            'Tajikistan', 'Turkmenistan', 'Estonia', 'Uzbekistan']
    moscow_region = ['Aprelevka', 'Balashikha', 'Chekhov', 'Chernogolovka', 'Dedovsk',
                     'Dmitrov', 'Dolgoprudny', 'Domodedovo', 'Dubna', 'Dzerzhinsky',
                     'Elektrogorsk', 'Elektrostal', 'Elektrougli', 'Fryazino', 'Golitsyno',
                     'Istra', 'Ivanteyevka', 'Kalininets', 'Kashira', 'Khimki', 'Khotkovo',
                     'Klimovsk', 'Klin', 'Kolomna', 'Korolyov', 'Kotelniki', 'Krasnoarmeysk',
                     'Krasnogorsk', 'Krasnoznamensk', 'Kubinka', 'Kurovskoye',
                     'Likino-Dulyovo', 'Lobnya', 'Losino-Petrovsky', 'Lukhovitsy',
                     'Lytkarino', 'Lyubertsy', 'Mozhaysk', 'Mytishchi', 'Naro-Fominsk',
                     'Noginsk', 'Odintsovo', 'Orekhovo-Zuyevo', 'Pavlovsky Posad', 'Podolsk',
                     'Protvino', 'Pushchino', 'Pushkino', 'Ramenskoye', 'Reutov', 'Ruza',
                     'Sergiyev Posad', 'Serpukhov', 'Shatura', 'Shchyolkovo',
                     'Solnechnogorsk', 'Staraya Kupavna', 'Stupino', 'Vidnoye',
                     'Volokolamsk', 'Voskresensk', 'Yakhroma', 'Yegoryevsk', 'Zvenigorod']
    millionaire_cities = ['Kazan', 'Moscow', 'Yekaterinburg', 'Saint Petersburg', 'Novosibirsk', 'Nizhny Novgorod',
                          'Chelyabinsk', 'Omsk', 'Samara', 'Ufa', 'Rostov-on-Don', 'Volgograd', 'Krasnodar', 'Perm']

    if 'session_id' in data.columns:
        data = data.set_index('session_id')

    if 'visit_date' in data.columns:
        data['visit_date'] = data['visit_date'].astype('datetime64[ns]')
        data['weekday'] = data['visit_date'].dt.weekday.astype('category')
        data['visit_day'] = data['visit_date'].dt.day.astype('category')

    if 'visit_time' in data.columns:
        data['visit_time'] = data['visit_time'].astype(str)
        data['visit_time'] = data['visit_time'].astype('datetime64[ns]')
        data['hour'] = data.apply(lambda x: x['visit_time'].hour, axis=1).astype('category')
        data['night'] = data['visit_time'].dt.hour < 8

    if 'utm_medium' in data.columns:
        data['is_organic'] = data['utm_medium'].isin(organic)

    if 'utm_source' in data.columns:
        data['is_social'] = data['utm_source'].isin(social)

    if 'device_screen_resolution' in data.columns:
        data[['screen_width', 'screen_height']] = data['device_screen_resolution'].str.split('x', expand=True).astype(
            float)
        data['screen_ratio'] = data['screen_width'] / data['screen_height']
        data['screen_area'] = data['screen_width'] * data['screen_height']

    if 'geo_country' in data.columns:
        data['is_ussr'] = data['geo_country'].isin(ussr)

    if 'geo_city' in data.columns:
        data['moscow_region'] = data['geo_city'].isin(moscow_region)
        data['millionaire_city'] = data['geo_city'].isin(millionaire_cities)
        data['distance_from_moscow'] = data['geo_city'].apply(lambda x: distance(x))

    return data


def convert_float(data: pd.DataFrame) -> pd.DataFrame:
    return data.astype(float)


def get_metadata(model, df_full):

    X_train, X_test, y_train, y_test = train_test_split(df_full.drop(columns=['event_value']),
                                                        df_full['event_value'],
                                                        test_size=0.15,
                                                        stratify=df_full['event_value'],
                                                        random_state=42)
    model.fit(X_train, y_train)
    test_proba = model.predict_proba(X_test)[:, 1]
    final_threshold = find_threshold(y_test, test_proba)
    test_prediction = (test_proba > final_threshold).astype(int)

    return {
        'name': 'СберАвтоподписка',
        'description': 'Модель для определения совершения целевого действия',
        'model': model['model'].__class__.__name__,
        'version': 1.0,
        'model_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'author': 'Sergey Tuktarov',
        'metrics': {
            'threshold': final_threshold,
            'roc_auc': round(roc_auc_score(y_test, test_prediction), 2),
            'accuracy': round(accuracy_score(y_test, test_prediction), 2),
            'precision': round(precision_score(y_test, test_prediction), 2),
            'recall': round(recall_score(y_test, test_prediction), 2),
            'f1': round(f1_score(y_test, test_prediction), 2),
               }
    }


def find_threshold(y: pd.Series, y_proba: pd.Series, best_threshold: float = 0.5) -> float:
    def roc_auc(threshold: float) -> float:
        predict = (y_proba > threshold).astype(int)
        return roc_auc_score(y, predict)

    best_metric = roc_auc(best_threshold)
    direction = 1
    shift = 0.2

    for i in range(300):
        threshold = best_threshold + direction * shift
        shift *= 0.9
        metric = roc_auc(threshold)
        if metric > best_metric:
            best_metric = metric
            best_threshold = threshold
        else:
            direction *= -1
    return best_threshold


def save_model(model: BaseEstimator, metadata: dict) -> None:
    filename = f'model_{datetime.now():%Y%m%d%H%M%S}.pkl'
    model.metadata = metadata
    with open(f'models/{filename}', 'wb') as file:
        dill.dump(model, file)


if __name__ == '__main__':
    model_creation()