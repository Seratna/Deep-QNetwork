import json
import socket

import requests


class DataManager(object):
    """

    """
    def __init__(self):
        pass

    def get_prices(self, ticker):
        """

        """
        years = [('2017-01-01', '2017-03-05')] + \
                [('{}-01-01'.format(y), '{}-12-31'.format(y)) for y in range(2016, 1999, -1)]

        quotes = []
        for start_date, end_date in years:
            query = 'select * from yahoo.finance.historicaldata where ' \
                    'symbol = "{}" and startDate = "{}" and endDate = "{}"'.format(ticker, start_date, end_date)
            params = {'q': query,
                      'format': 'json',
                      'env': 'store://datatables.org/alltableswithkeys',
                      'callback': ''}
            url = 'https://query.yahooapis.com/v1/public/yql'
            while True:
                timeout = False
                try:
                    r = requests.get(url, params=params, timeout=(3.05, 3.05))
                except requests.exceptions.Timeout as e:
                    print(e)
                    timeout = True
                except Exception as e:
                    print(e)
                    print('type:', type(e))
                    timeout = True

                if (not timeout) and r:
                    break

            ans = r.json()
            if ans['query']['count'] > 0:
                quotes += ans['query']['results']['quote']

        with open('data/{}.json'.format(ticker), 'w') as file:
            json.dump(quotes, file)

    def download(self, companies_file):
        with open(companies_file) as file:
            companies = json.load(file)
        for i, company in enumerate(companies):
            if i > 2100:
                symbol = company['Symbol']
                print('downloading data of', symbol, '{}/{}'.format(i+1, len(companies)))

                self.get_prices(symbol)


def main():
    dm = DataManager()
    dm.download('companies.json')


if __name__ == '__main__':
    main()
