import os
import requests
import fake_useragent
from bs4 import BeautifulSoup
import time
import json
import random
import pandas as pd
from datetime import datetime


def ser_links():
    links = ['https://www.imdb.com/chart/toptv/', 'https://www.imdb.com/chart/tvmeter/']
    serials_links = set()
    for link in links:
        user = fake_useragent.UserAgent().random
        header = {'user-agent': user}
        response = requests.get(link, headers=header)
        soup = BeautifulSoup(response.content, 'html.parser')
        json_ld = soup.find('script', type='application/ld+json')
        data = json.loads(json_ld.string)
        for item in data['itemListElement']:
            series = item['item']
            serials_links.add(series.get('url'))
        time.sleep(2)
    serials_links = list(serials_links)
    return serials_links


def serials_info(serials_links):
    serials_info = []
    for link in serials_links:
        user = fake_useragent.UserAgent().random
        header = {'user-agent': user}
        response = requests.get(link, headers=header)
        if response.status_code != 200:
            print(response.status_code)
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        json_ld = soup.find('script', type='application/ld+json')
        data = json.loads(json_ld.string)
        serial_info = {
            'Title': data.get('name'),
            'Description': data.get('description'),
            'Genre': ', '.join(data.get('genre', [])),
            'Release Year': data.get('datePublished', '').split('-')[0],
            'Keywords': data.get('keywords'),
            'Actors': [actor['name'] for actor in data.get('actor', [])]}
        serials_info.append(serial_info)
        time.sleep(random.uniform(5, 15))
    return pd.DataFrame(serials_info)


def parser():
    links = ser_links()
    df = serials_info(links)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join('/app/data/', f'movies_{current_time}.csv')
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    parser()