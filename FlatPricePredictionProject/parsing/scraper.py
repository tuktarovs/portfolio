import requests
import time
import pandas as pd
import os
from datetime import datetime

def fetch_data(page):
    cookies = {
        '_gcl_au': '1.1.1282998986.1740565241',
        'cf_clearance': 'h0jw3iYPPduaKYDSQ6gXRghAPrIDHv_ZHuz9weykkgY-1740580831-1.2.1.1-JgKRrN29oCofr1DrwdMiYxe8wjJ9LhEb2SyFVYHuenVtljrHQ_mkbCWfJFytdgBARJfImtphvlVLGvVq6KuVvz09Pay_FIQVgSjhCX61dxkxvTAvHqyO6Y3wIpFLSGxuji7_EdZpiwsjUAg4zNGOenrg63PfgibfKh8ID7lQqHq6Bo_ofnfCRliwCAfnC_D3wZOJj5kPcNPcQOsXxgbi3NiRZUXuGWU3pmnOcN1exMB1HozOUM3mODHTJWx0zPWo.kCzC4X7GwVyMJoy60b6cQHSGiDX4lKgA3qiax3xdbQ',
        'tmr_lvid': '388635b3198e19bf5a858c6bb834b47b',
        'tmr_lvidTS': '1740565242343',
        'session_main_town_region_id': '1',
        'session_region_id': '1',
        '__cf_bm': '46JzEDPVUvUKzSVYf37BXEVG7RkRIiqkeYuWdnvrXSY-1740580134-1.0.1.1-16rEri6Mr3dZAXycCmroveHRGKC57bkkaOjBSENiYrXCm1s54pYdahEV_tik.uX11Yj0dd_fkFhFOcGL77hhpg',
        '_ym_visorc': 'b',
        'adrcid': 'AMnXWmYG2Pgn4cvqVi_N3gw',
        'afUserId': 'ba47cc07-3bd3-4abd-9213-3877fe1815c5-p',
        '_ga': 'GA1.1.920498561.1740565327',
        '_ga_3369S417EL': 'GS1.1.1740579630.3.1.1740579871.59.0.0',
        'sopr_session': 'fcff4c71dc9a450d',
        'cookie_agreement_accepted': '1',
        'transport-accessibility_onboarding_counter': '1',
        'uxfb_card_satisfaction': '%5B314008663%2C305062355%2C308091340%5D',
        'acs_3': '%7B%22hash%22%3A%222519d36ba1d6b3a4bd08e045fbf175fd06f869ed%22%2C%22nextSyncTime%22%3A1740651732046%2C%22syncLog%22%3A%7B%22224%22%3A1740565332046%2C%221228%22%3A1740565332046%7D%7D',
        'AF_SYNC': '1740565331551',
        'adrdel': '1740565331838',
        '_ym_isad': '2',
        'uxs_uid': '89673c30-f42b-11ef-817f-392771c822b2',
        '_ym_d': '1740565329',
        '_ym_uid': '1740565329848896824',
        'sopr_utm': '%7B%22utm_source%22%3A+%22direct%22%2C+%22utm_medium%22%3A+%22None%22%7D',
        'login_mro_popup': '1',
        '_CIAN_GK': 'fb3e4139-0f65-4b87-bb26-cb2ac6894c09', }

    headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Sec-Fetch-Site': 'same-site',
        'Accept-Language': 'ru',
        'Sec-Fetch-Mode': 'cors',
        'Origin': 'https://www.cian.ru',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15',
        'Referer': 'https://www.cian.ru/',
        'Sec-Fetch-Dest': 'empty',
        'Priority': 'u=3, i', }

    json_data = {
        'jsonQuery': {
            '_type': 'flatsale',
            'engine_version': {
                'type': 'term',
                'value': 2, },
            'region': {
                'type': 'terms',
                'value': [1, ], },
            "page": {
                "type": "term",
                "value": page},
            'room': {
                'type': 'terms',
                'value': [1, 2, ], },
            'building_status': {
                'type': 'term',
                'value': 1, }, }, }
    response = requests.post(
        'https://api.cian.ru/search-offers/v2/search-offers-desktop/',
        cookies=cookies,
        headers=headers,
        json=json_data, )
    return response


def get_offers(offers, data):
    for item in data:
        offer = {}
        offer['id'] = item['id']
        offer['floor'] = item['floorNumber']
        offer['price'] = item['bargainTerms']['priceRur']
        offer['rooms_count'] = item['roomsCount']
        offer['total_area'] = item['totalArea']
        offer['kitchen_area'] = item['kitchenArea']
        offer['year'] = item['building']['buildYear']
        try:
            offer['parking'] = item['building']['parking']['type']
        except:
            offer['parking'] = None
        offer['total_floors'] = item['building']['floorsCount']
        offer['lat'] = item['geo']['coordinates']['lat']
        offer['lng'] = item['geo']['coordinates']['lng']
        try:
            offer['balconiesCount'] = item['balconiesCount']
        except:
            offer['balconiesCount'] = 0
        offer['materialType'] = item['building']['materialType']
        offers.append(offer)


def parse_data():
    offers = []
    page = 1
    while True:
        response = fetch_data(page)
        if response.status_code == 200:
            data = (response.json())['data']['offersSerialized']
            if data == []:
                break
            else:
                get_offers(offers, data)
            print(f'Сбор данных со страницы {page} завершен')
            print(f'Собрано {len(offers)} квартир')
            page += 1
        else:
            print(response.status_code)
            time.sleep(15)
        time.sleep(20)
    df = pd.DataFrame(offers)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join('..', 'data', 'raw', f'offers_{current_time}.csv')
    df.to_csv(file_path, index=False)
    print(f'Данные сохранены в {file_path}')