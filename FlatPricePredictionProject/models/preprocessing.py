import pandas as pd
import numpy as np


def missings(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    missing_values = [float('nan'), '(none)']
    if 'balconiesCount' in data.columns:
        data['balconiesCount'] = data['balconiesCount'].fillna(0)
    if 'kitchen_area' in data.columns:
        data['kitchen_area'] = data['kitchen_area'].replace(missing_values, data.kitchen_area.mode()[0])

    if 'year' in data.columns:
        data['year'] = data['year'].replace(missing_values, data.year.mode()[0])

    if 'floor' in data.columns:
        data['floor'] = data['floor'].replace(missing_values, data.floor.mode()[0])

    if 'rooms_count' in data.columns:
        data['rooms_count'] = data['rooms_count'].replace(missing_values, data.rooms_count.mode()[0])
    if 'total_floors' in data.columns:
        data['total_floors'] = data['total_floors'].replace(missing_values, data.total_floors.mode()[0])
    if 'distance_to_center' in data.columns:
        data['distance_to_center'] = data['distance_to_center'].replace(missing_values, data.distance_to_center.mode()[0])
    return data.fillna('Unknown')


def new_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    def haversine(lat, lon):
        R = 6371.0
        moscow_lat = 55.7558
        moscow_lng = 37.6173
        lat1_rad = np.radians(moscow_lat)
        lon1_rad = np.radians(moscow_lng)
        lat2_rad = np.radians(lat)
        lon2_rad = np.radians(lon)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        return distance

    def determine_floor_status(row):
        if row['floor'] == row['total_floors']:
            return 'last_floor'
        elif row['floor'] == 1:
            return 'first_floor'
        else:
            return 'normal_floor'

    if 'lat' in data.columns and 'lng' in data.columns:
        data['distance_from_center'] = data.apply(lambda row: haversine(row['lat'], row['lng']), axis=1)
    if 'floor' in data.columns and 'total_floors' in data.columns:
        data['floor_status'] = data.apply(determine_floor_status, axis=1).astype('category')
    return data