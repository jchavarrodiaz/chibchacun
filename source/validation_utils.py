from conn_presto import get_stations_by_sensor, get_sensor_station_data
from db_utils import get_config_params
import prestodb
import pandas as pd
import scipy.stats as ss


def format_df_data(df_input, station='Data'):
    name_cols = ['Station', 'Sensor', 'Date', 'Data']
    df_input.columns = name_cols
    df_input.drop(['Station', 'Sensor'], axis=1, inplace=True)
    df_input['Date'] = pd.to_datetime(df_input['Date'], infer_datetime_format=True)
    df_input.reset_index(inplace=True, drop=True)
    df_input.set_index(['Date'], inplace=True)
    df_input.sort_index(inplace=True)
    sr_output = df_input['Data']
    sr_output.name = str(station)
    return sr_output


def calc_freq(sr_data):
    """
    Returns time series frequency in minutes
    :param sr_data: series to extract freq
    :return: frequency in minutes
    """
    sr_diff = sr_data.index.to_series().diff() / pd.Timedelta('1M')
    sr_all_freqs = sr_diff.value_counts(normalize=True)
    sr_all_freqs.name = sr_data.name

    if sr_all_freqs.iloc[0] < .96:
        print('\n')
        print(sr_all_freqs.iloc[:5])

    freq = sr_diff.mode()[0]
    # freq /= pd.Timedelta('1M')

    return int(freq)


def get_elevations():
    """
    Returns elevations for a set of stations based on a DEM.

    :return:
    """
    params = get_config_params('../config/database.ini', section='prestodb_cassandra')
    conn = prestodb.dbapi.Connection(**params)
    query_table = 'last_month_observations'
    sensor = '0240'

    stations = get_stations_by_sensor(conn=conn, table=query_table, sensor=sensor)
    stations = [item for sublist in stations for item in sublist]
    stations.sort()

    df_summary = pd.DataFrame(index=stations, columns=['Freq', 'Start', 'End', 'Count'])
    df_summary.index.name = 'Station'

    for station in stations:
        # print(station)
        results = get_sensor_station_data(station=station, sensor=sensor, conn=conn, table=query_table, write_csv=False)
        df_station = pd.DataFrame(results)
        sr_station = format_df_data(df_input=df_station, station=station)
        freq_min = calc_freq(sr_station)
        start = sr_station.index.min()
        end = sr_station.index.max()
        count = sr_station.dropna().count()
        df_summary.loc[station] = [freq_min, start, end, count]

    df_summary.to_excel('../results/elevations_cassandra.xlsx')
    return df_summary


def grubbs_test(df_input, alpha=0.05, two_tail=True):
    """
    This function applies the Grubbs' Test for outliers in a dataframe and returns two dataframes, the first one
    without outliers and the second one just for the outliers
    :param df_input: Pandas dataframe with series to test.
    :param alpha: Significance level [1% as default].
    :param two_tail: Two tailed distribution [True as default].
    :return: tuple with two dataframes, the first one without outliers and the second one just for outliers.
    """

    if isinstance(df_input, pd.Series):
        df_input = pd.DataFrame(df_input)

    df_try = df_input.copy()
    df_output = pd.DataFrame(index=df_input.index, columns=df_input.columns)
    df_outliers = pd.DataFrame(data=0, index=df_input.index, columns=df_input.columns)

    if two_tail:
        alpha /= 2

    i = 0
    while not df_outliers.isnull().values.all():
        mean = df_try.mean()
        std = df_try.std()
        n = len(df_try)
        tcrit = ss.t.ppf(1 - (alpha / (2 * n)), n - 2)
        zcrit = (n - 1) * tcrit / (n * (n - 2 + tcrit ** 2)) ** .5
        df_outliers = df_try.where(((df_try - mean) / std).abs() > zcrit)
        df_output.update(df_input[df_outliers.isnull() == False])
        # df_output.update(df_input[df_outliers.isnotnull()])
        df_try = df_try[df_outliers.isnull()]
        i += 1

    return df_try, df_output


if __name__ == '__main__':
    get_elevations()
    pass
