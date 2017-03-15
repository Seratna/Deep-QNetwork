from stockex import stockwrapper as sw


def main():
    data = sw.YahooData()
    print(data.get_historical('GOOG', startDate='2009-01-01'))


if __name__ == '__main__':
    main()
