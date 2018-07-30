from configparser import ConfigParser
import os
import datetime
import pandas as pd
from dateutil import tz


def make_dir(path):
    """
    Makes directory based on path.

    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_paths(path, current_time=None, products=None, periods=None):
    """
    Makes paths for saving products.

    :param path:
    :param current_time:
    :param products:
    :param periods:
    :return:
    """

    to_zone = tz.gettz('UTC')
    current_time = current_time.astimezone(to_zone)

    year = current_time.year
    month = current_time.month
    day = current_time.day

    if products is None:
        products = ['tif', 'png', 'csv', 'xls']

    if periods is None:
        periods = ['1D']

    [[make_dir('{}/{}/{:04}/{:02}/{:02}/{:0>3}'.format(path, i, year, month, day, j))
      for j in periods] for i in products]


def get_pars_from_ini(filename='../config/zones.ini'):
    """
    Returns dictionary with zone configuration for interpolating.

    :param filename: .ini file name
    :type filename: str
    :return:
    """
    parser = ConfigParser()
    parser.read(filename)

    dt_pars = {}

    zones = parser.sections()

    for zone in zones:
        db = {}
        params = parser.items(zone)

        for param in params:
            try:
                db[param[0]] = eval(param[1])

            except ValueError:
                db[param[0]] = param[1].strip()

            except NameError:
                db[param[0]] = param[1].strip()

            except SyntaxError:
                db[param[0]] = param[1].strip()

        dt_pars[zone] = db

    return dt_pars


def build_product_name(date, product='P', sensor='STA', zone='COL', backward='1D', ret_path=False):
    """
    Returns product name based on date and other parameters.

    :param date: date-time of the end of the period.
    :type date: Timestamp
    :param product: Product abbreviation. P: Precipitation, T: Temperature.
    :type product: str
    :param sensor: Sensor abbreviation. STA: Station, RAD: Radar, SAT: Satellite.
    :type sensor: str
    :param zone: Zone abbreviation, it depends on the product.
    :type zone: str
    :param backward: Backward period in minutes, hours or days. H: Hours, M: Minutes, D: Days.
    :type backward: str
    :param ret_path: returns path file based on date?
    :type ret_path: False
    :return: Filename for Product.
    :rtype: str
    """

    to_zone = tz.gettz('UTC')
    date = date.astimezone(to_zone)

    filename = '{}_{}_{}_{:%Y%m%d%H%M}_{:0>3}'.format(product, sensor, zone, date, backward)

    if ret_path:
        path = '{:%Y/%m/%d}/{:0>3}'.format(date, backward)
        filename = '{}/{}'.format(path, filename)

    return filename


def date_start_end(backward_period='1H', current_time=None, timezone='America/Bogota', delay='1H', freq='1H',
                   lag=None):
    """
    Returns start and end time in UTC for a query based on current date-time and a backward period.

    For backwards periods see time conversions https://docs.python.org/2/library/time.html.
    1 Hour: 1H, 3 Hours: 3H, 1 Meteorological Day: 1D, etc.
    :param backward_period: Backward period, for meteorological days use D.
    :type backward_period: str
    :param current_time: time for calculate times.
    :type current_time: Timestamp
    :param timezone: time zone for current time.
    :type timezone: str
    :param delay: Delay for taking into account data, for stations is 1 Hour.
    :type delay: str
    :param freq: Product frequency, apply only for H backwards. (radar: '5M', stations:'1H', satellite: '30M')
    :type freq: str
    :param lag: Product Lag, apply only for H backwards. (satellite: '15M')
    :type lag: str
    :return:
    :rtype: Timestamp, Timestamp
    """

    if current_time is None:
        current_time = datetime.datetime.now()

    from_zone = tz.gettz(timezone)
    to_zone = tz.gettz('UTC')
    current_time = current_time.replace(microsecond=0, second=0, tzinfo=from_zone)

    if timezone != 'UTC':
        current_time = current_time.astimezone(to_zone)

    if backward_period[-1] == 'D':
        time_query = current_time - pd.Timedelta(delay)

        if time_query.hour < 12:
            time_query = time_query - pd.Timedelta('1D')
            time_query = time_query.replace(hour=12, minute=0)

        else:
            time_query = time_query.replace(hour=12, minute=0)

    else:
        time_query = current_time - pd.Timedelta(delay)

        if lag is None:
            _lag = 0

        else:
            _lag = int(pd.Timedelta(lag) / '1M')

        _freq = int(pd.Timedelta(freq) / '1M')
        minutes = range(_lag, 60, _freq)
        minute_query = time_query.minute

        if minute_query < minutes[0]:
            time_query = time_query - pd.Timedelta('1H')
            time_query = time_query.replace(minute=minutes[-1])

        elif minute_query >= minutes[-1]:
            time_query = time_query.replace(minute=minutes[-1])

        else:
            last_minute_available = max([i for i in minutes if i <= minute_query])
            time_query = time_query.replace(minute=last_minute_available)

    end_time = time_query
    start_time = end_time - pd.Timedelta(backward_period)

    if timezone != 'UTC':
        start_time = start_time.astimezone(from_zone)
        end_time = end_time.astimezone(from_zone)

    return start_time, end_time


if __name__ == '__main__':
    pass
