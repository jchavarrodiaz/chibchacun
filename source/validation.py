from conn_presto import get_by_query_presto
from db_utils import get_config_params
import prestodb
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def get_data(station, sensor, table='last_month_observations'):
    """
    Returns the whole data from a given table for the sensor and station given.

    :param station:
    :type station: str
    :param sensor:
    :type sensor: str
    :param table:
    :type table: str
    :return:
    """
    catalog = 'cassandra'
    section = 'prestodb_{}'.format(catalog)
    params = get_config_params('../config/database.ini', section=section)
    conn = prestodb.dbapi.Connection(**params)

    sql = """
    SELECT to_char(event_time + interval '5' hour,'yyyy-mm-dd hh24:mi') event_time, event_value
    FROM {0}
    WHERE station = '{1}'
    AND sensor = '{2}'
    ORDER BY event_time
    """.format(table, station, sensor)

    results = get_by_query_presto(conn=conn, sql=sql)
    name_cols = ['Time', station]

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        df_results.columns = name_cols
        df_results['Time'] = pd.to_datetime(df_results['Time'], format='%Y-%m-%d %H:%M')
        df_results.set_index('Time', inplace=True)

        df_results[station].to_pickle('../test/{}_{}_lm.pkl'.format(sensor, station))


def apply_boxplot(sr_input):
    """
    Apply box-plot calculations for a series.

    Returns max and min values for 1.5 (suspect) and 3.0 (outlier) times Inter-Quartile Range.
    :param sr_input:
    :type sr_input: pd.Series or pd.DataFrame
    :return:
    :rtype: pd.DataFrame or pd.Series
    """
    sr_qx = sr_input.quantile([.25, .5, .75])
    q1 = sr_qx.loc[.25]
    q3 = sr_qx.loc[.75]
    iqr = q3 - q1

    min_suspect = q1 - 1.5 * iqr
    max_suspect = q3 + 1.5 * iqr
    min_outlier = q1 - 3. * iqr
    max_outlier = q3 + 3. * iqr

    try:
        df_limits = pd.DataFrame({'min_suspect': min_suspect, 'max_suspect': max_suspect,
                                  'min_outlier': min_outlier, 'max_outlier': max_outlier}).T

    except ValueError:
        df_limits = pd.Series({'min_suspect': min_suspect, 'max_suspect': max_suspect,
                               'min_outlier': min_outlier, 'max_outlier': max_outlier})

    return df_limits


def min2hour(sr_input):
    """
    Aggregates minute time series to hourly.

    Returns a dataframe with
    :param sr_input:
    :type sr_input: pd.Series
    :return:
    :rtype: pd.DataFrame
    """
    sr_input = sr_input.resample('1H').mean()
    df_input = pd.DataFrame(sr_input)
    df_input['Date'] = df_input.index.date
    df_input['Hour'] = df_input.index.hour
    df_output = df_input.pivot(index='Date', columns='Hour', values=df_input.columns[0])

    return df_output


def main():
    station = '0029004520'
    sensor = '0068'
    get_data(station=station, sensor=sensor)

    sr_station = pd.read_pickle('../test/{}_{}_lm.pkl'.format(sensor, station))
    plot_acf(sr_station)
    plt.show()
    plt.close()
    sr_lim_q1 = apply_boxplot(sr_station)
    print("\nQ2 Test:\n{}".format(sr_lim_q1.to_string()))
    sr_station.plot.box()
    plt.show()
    plt.close()

    hg_station = min2hour(sr_station)  # Hourly Group
    df_limits = apply_boxplot(hg_station)
    print("\nQ3 Test:\n{}".format(df_limits.to_string()))
    hg_station.plot.box(notch=False, sym='.', whis=1.5)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
    pass
