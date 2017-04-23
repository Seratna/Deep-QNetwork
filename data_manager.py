import json
import socket

import requests
from requests import ConnectionError, Timeout,  HTTPError


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
                except (ConnectionError, Timeout,  HTTPError) as e:
                    print(e)
                    timeout = True
                except Exception as e:
                    print(e)
                    print('type:', type(e))
                    timeout = True

                if (not timeout) and r:
                    break

            ans = r.json()
            count = ans['query']['count']

            if count > 0:
                if count == 1:
                    quotes.append(ans['query']['results']['quote'])
                else:
                    quotes.extend(ans['query']['results']['quote'])

            quotes.reverse()

        with open('prices/{}.json'.format(ticker), 'w') as file:
            json.dump(quotes, file)

    def download(self, companies_file):
        with open(companies_file) as file:
            companies = json.load(file)
        for i, company in enumerate(companies):
            symbol = company['Symbol']
            print('downloading data of', symbol, '{}/{}'.format(i+1, len(companies)))

            self.get_prices(symbol)


def main():
    dm = DataManager()
    dm.download('companies.json')


if __name__ == '__main__':
    main()
