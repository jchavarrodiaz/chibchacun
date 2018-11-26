import datetime
from functools import partial
from multiprocessing import Pool

import pandas as pd

from inter_precipitation import interpolate_pt
from inter_temperature import interpolate_ts
from inter_utils import dt_sensors_abb


def run_hourly(current_time=None, multiprocessing=False):
    """
    Runs hourly precipitation interpolation.
    :param multiprocessing:
    :param current_time:
    :return:
    """

    if current_time is None:
        current_time = datetime.datetime.now()

    start_inter_time = datetime.datetime.now()
    print("Inicio interpolacion horaria: {:%Y-%m-%d %H:%M}".format(start_inter_time))
    backward_periods = ['6H']  # ['1H', '2H', '3H', '6H', '12H', '24H']
    zones = ['Bogota']  # , 'Colombia']
    sensors = ['0240']

    for sensor in sensors:
        sensor_abb = dt_sensors_abb[sensor]

        for zone in zones:
            print(sensor, zone)

            if sensor_abb == 'PT':
                partial_function = partial(interpolate_pt, zone=zone, sensor=sensor, current_time=current_time)

            elif sensor_abb == 'TS':
                partial_function = partial(interpolate_ts, zone=zone, sensor=sensor, current_time=current_time)

            else:
                continue

            if multiprocessing:
                pool = Pool()
                pool.map(partial_function, backward_periods)
                pool.close()

            else:
                map(partial_function, backward_periods)

    end_inter_time = datetime.datetime.now()
    print("Fin interpolacion horaria: {:%Y-%m-%d %H:%M}".format(end_inter_time))


def run_daily(multiprocessing=True, current_time=None):
    start_inter_time = datetime.datetime.now()
    print("Inicio interpolacion diaria: {:%Y-%m-%d %H:%M}".format(start_inter_time))

    # run_daily_ts(multiprocessing=multiprocessing, current_time=current_time)
    run_daily_pt(multiprocessing=multiprocessing, current_time=current_time)

    end_inter_time = datetime.datetime.now()
    print("Fin interpolacion diaria: {:%Y-%m-%d %H:%M}".format(end_inter_time))


def run_daily_ts(multiprocessing=True, current_time=None):
    """
    Runs daily temperature interpolation.

    :param multiprocessing:
    :param current_time:
    :return:
    """

    if current_time is None:
        current_time = datetime.datetime.now()

    backward_periods = ['1D', '2D', '3D', '7D', '10D', '30D']
    zones = ['Bogota', 'Colombia']
    sensors = ['0069', '0070']

    for sensor in sensors:

        for zone in zones:
            print(sensor, zone)
            partial_function = partial(interpolate_ts, zone=zone, sensor=sensor, current_time=current_time)

            if multiprocessing:
                pool = Pool()
                pool.map(partial_function, backward_periods)
                pool.close()

            else:
                map(partial_function, backward_periods)


def run_daily_pt(multiprocessing=True, current_time=None):
    """
    Runs daily precipitation interpolation.
    :param multiprocessing:
    :param current_time:
    :return:
    """

    if current_time is None:
        current_time = datetime.datetime.now()

    backward_periods = ['1D', '2D', '3D', '7D', '10D']
    zones = ['Bogota', 'Colombia']
    sensor = '0240'

    for zone in zones:
        print(sensor, zone)

        partial_function = partial(interpolate_pt, zone=zone, current_time=current_time)

        if multiprocessing:
            pool = Pool()
            pool.map(partial_function, backward_periods)
            pool.close()

        else:
            map(partial_function, backward_periods)


def multiple_days(multiprocessing=False):
    start = '2018-07-30 09:50'
    end = '2018-08-29 09:50'
    dates = pd.date_range(start=start, end=end, freq='D')

    for date in dates:
        print(date)
        run_daily(multiprocessing=multiprocessing, current_time=date)


def multiple_hours(multiprocessing=False):
    start = '2018-09-30 19:20'
    end = '2018-11-25 19:20'
    dates = pd.date_range(start=start, end=end, freq='6H')

    if multiprocessing:
        pool = Pool()
        pool.map(run_hourly, dates)
        pool.close()

    else:
        map(run_hourly, dates)


def run_by_demand(current_time=None, sensor='0240', backward_periods='1D', zone='Colombia', multiprocessing=True):
    """
    Runs an interpolation by demand.

    :param current_time: Query time.
    :type current_time: str
    :param sensor: Sensor code.
    :type sensor: str
    :param backward_periods: Comma separated backward periods
    :type backward_periods: str
    :param zone: Interpolation zone
    :type zone: str
    :param multiprocessing:
    :type multiprocessing: bool
    :return:
    """

    if current_time is None:
        current_time = datetime.datetime.now()

    elif isinstance(current_time, str):
        current_time = pd.to_datetime(current_time)

    if backward_periods is None:
        print('No se ingresaron periodos precedentes.')
        exit()

    elif isinstance(backward_periods, str):
        backward_periods = [i.strip().upper()
                            for i in backward_periods.split(',')
                            if ('H' in i.upper()) or ('D' in i.upper())]

    if sensor == '0240':
        partial_function = partial(interpolate_pt, zone=zone, current_time=current_time)

    elif sensor == '0069' or sensor == '0070':
        partial_function = partial(interpolate_ts, zone=zone, sensor=sensor, current_time=current_time)

    else:
        print("No se escogio un sensor valido.")
        partial_function = None
        exit()

    if multiprocessing:
        pool = Pool()
        pool.map(partial_function, backward_periods)
        pool.close()

    else:
        map(partial_function, backward_periods)


def main():
    current_time = pd.to_datetime('2018-05-23 10:05')
    run_daily(True, current_time=current_time)


if __name__ == '__main__':
    # multiple_days(True)
    multiple_hours(True)
    pass
