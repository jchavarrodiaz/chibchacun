import csv
import sys
from functools import partial
from multiprocessing import Pool

import pandas as pd
import prestodb

from db_utils import get_config_params

"""
Ejemplo de una consulta con agrupamiento de informacion

SELECT to_char(event_time + interval '5' hour,'yyyy-mm-dd hh') fecha, 
sum(event_value) event_value_aggr, 
count(*) cantidad
FROM raw.last_month_observations
WHERE station='0021195170' AND sensor='0240'
AND to_char(event_time + interval '5' hour,'yyyy-mm')='2017-09'
AND event_value > 0 AND event_value < 20
GROUP BY to_char(event_time + interval '5' hour,'yyyy-mm-dd hh')
HAVING sum(event_value) > 1
ORDER by 1
"""

reload(sys)
sys.setdefaultencoding('utf-8')


dt_fields = {'cassandra': {'station': 'station', 'sensor': 'sensor', 'time': 'event_time', 'value': 'event_value'},
             'postgresql': {'station': 'id_stz', 'sensor': 'id_measure', 'time': 'date_record', 'value': 'raw_data'}}
dt_sensors_postgres = {'0240': 1, '0068': 5, '0069': 85, '0070': 45, '0230': 7,
                       '0255': 4, '0103': 2, '0104': 3, '0027': 6, '0239': 12}


def get_by_query_presto(conn, sql):
    """
    Query database by sql sentence.
    :param conn:
    :param sql:
    :return:
    """
    try:
        curs = conn.cursor()
        curs.execute(sql)
        results = curs.fetchall()
        return results

    except prestodb.exceptions.PrestoUserError, e:
        print(e)


def get_all_data(conn, table='stations'):
    """
    Gets all data from a table. It must be used with careful because it could block server.
    :param conn:
    :param table:
    :return:
    """
    cur = conn.cursor()
    query = "SELECT * FROM {}".format(table)
    cur.execute(query)
    results = cur.fetchall()
    header = [i[0] for i in cur.description]
    results.insert(0, header)

    return results


def summary_sensors_stations(catalog='cassandra', table_data='last_month_observations'):
    """
    Makes a summary of stations and sensors in order to identify what sensors are
    recording in each station.

    :param table_data:
    :param catalog:
    :type table_data: str
    :type catalog: str
    :return:
    """
    section = 'prestodb_{}'.format(catalog)
    params = get_config_params('../config/database.ini', section=section)
    conn = prestodb.dbapi.Connection(**params)

    # query_sensors = get_all_data(conn, table='sensors')
    # df_sensors = pd.DataFrame(data=query_sensors[1:], columns=query_sensors[0])
    # df_sensors.set_index('sensorid', inplace=True)
    # sensors = df_sensors.index

    sensors = ['0240', '0068', '0069', '0070', '0230', '0255', '0103', '0104', '0027', '0239']

    if catalog == 'postgresql':
        sensors = [dt_sensors_postgres[i] for i in sensors]
        table_stations = 'configuration.stations'
        id_stations = 'id_stz'

    else:
        table_stations = 'raw.stations'
        id_stations = 'stationid'

    query_stations = get_all_data(conn, table=table_stations)
    df_stations = pd.DataFrame(data=query_stations[1:], columns=query_stations[0])
    df_stations.set_index(id_stations, inplace=True)
    stations = df_stations.index

    df_stations_sensors = pd.DataFrame(index=pd.Index(stations, name='Station'), columns=sensors)

    for sensor in sorted(sensors):
        print sensor
        results = get_stations_by_sensor(conn, table=table_data, sensor=sensor)
        list_stations = [i[0] for i in results]
        idx_miss = pd.Index(list_stations).difference(df_stations_sensors.index)

        for sta_miss in idx_miss:
            df_stations_sensors.loc[sta_miss] = float('NaN')

        df_stations_sensors.loc[list_stations, sensor] = 1

    xls_output = pd.ExcelWriter('../data/presto_{}_summary.xlsx'.format(catalog))
    # df_sensors.to_excel(xls_output, 'Sensors')
    df_stations.to_excel(xls_output, 'Stations')
    df_stations_sensors.to_excel(xls_output, 'Summary')
    xls_output.save()


def get_stations_by_sensor(conn, catalog='cassandra', table='weather_events', sensor='0240'):
    """
    Returns a list of stations that record data from a sensor.
    :param conn: Database connection
    :param catalog: Database catalog
    :param table: Table name
    :param sensor: Sensor code
    :return:
    """
    cur = conn.cursor()
    # catalog = conn._kwargs['catalog']

    if catalog == 'cassandra':
        val_sensor = "'{}'".format(sensor)

    else:
        val_sensor = sensor

    id_station = dt_fields[catalog]['station']
    id_sensor = dt_fields[catalog]['sensor']

    query = """
    SELECT distinct {0}
    FROM {1}
    WHERE {2}={3}
    """.format(id_station, table, id_sensor, val_sensor)

    cur.execute(query)
    return cur.fetchall()


def get_sensor_station_data(station, sensor, conn, catalog='cassandra', table='weather_events', write_csv=True):
    """
    Gets data from Cassandra table based on station and sensor, and
    write an csv file with data.
    :param station:
    :param sensor:
    :param conn:
    :param catalog:
    :param table:
    :param write_csv:
    :return:
    """
    # catalog = conn._kwargs['catalog']

    id_station = dt_fields[catalog]['station']
    id_sensor = dt_fields[catalog]['sensor']
    id_time = dt_fields[catalog]['time']
    id_value = dt_fields[catalog]['value']

    if catalog == 'cassandra':
        val_station = "'{:010}'".format(station) if not isinstance(station, basestring) else station
        val_sensor = "'{}'".format(sensor) if not isinstance(sensor, basestring) else sensor

    else:
        val_station = station
        val_sensor = sensor

    sql = """SELECT {0} station, {1} sensor, {2} event_date, {3} event_value 
    FROM {4} 
    WHERE {0}='{5}' 
    AND {1}='{6}'""".format(id_station, id_sensor, id_time, id_value, table, val_station, val_sensor)

    if write_csv:
        try:
            csv_file_dest = '../results/{0}/data{0}_{1}_{2}.csv'.format(catalog, sensor, station)
            outfile = open(csv_file_dest, 'w')  # 'wb'
            output = csv.writer(outfile, dialect='excel')

            curs = conn.cursor()
            curs.execute(sql)
            results = curs.fetchall()

            for row_data in results:  # add table rows
                output.writerow(row_data)

            outfile.close()

        except prestodb.exceptions.PrestoUserError, e:
            print(e)

    else:
        try:
            curs = conn.cursor()
            curs.execute(sql)
            results = curs.fetchall()
            return results

        except prestodb.exceptions.PrestoUserError, e:
            print(e)


def download_all_data(sensor='0240', table='weather_events', multiprocess=False):
    """
    Downloads all data from a table through a query based on sensor and
    stations from presto_data_summary.xlsx file.
    :param sensor:
    :param table:
    :param multiprocess:
    :return:
    """

    xls_stations = pd.ExcelFile('../data/presto_cassandra_summary.xlsx')
    df_summary = xls_stations.parse('Summary', index_col='Station')
    sr_0240 = df_summary[sensor]
    stations_0240 = sr_0240.dropna().index

    params = get_config_params('../config/database.ini', section='prestodb_cassandra')
    conn = prestodb.dbapi.Connection(**params)

    partial_fn = partial(get_sensor_station_data, sensor=sensor, conn=conn, table=table)

    if multiprocess:
        pool = Pool()
        pool.map(partial_fn, stations_0240)
        pool.close()

    else:
        map(partial_fn, stations_0240[:2])


def main():
    # sensors_stations()
    # list_stations = get_stations_by_sensor()
    # get_sensor_data()
    # get_stations()
    # sensors = ['{:04}'.format(i) for i in range(229, 236)]  # All stage (nivel) sensors
    sensors = ['0255', '0027']

    for sensor in sensors:
        download_all_data(sensor=sensor, multiprocess=True)


def download_from_stations(catalog='cassandra'):
    df_data = pd.read_excel('../data/extract_data.xlsx', sheetname='Sibate', index_col='COD_INTERNO')
    stations = df_data.index

    section = 'prestodb_{}'.format(catalog)
    params = get_config_params('../config/database.ini', section=section)
    conn = prestodb.dbapi.Connection(**params)

    for station in stations:
        get_sensor_station_data(station, sensor='0240', conn=conn)


if __name__ == '__main__':
    # summary_sensors_stations('postgresql', 'recent_data')
    summary_sensors_stations('cassandra', 'last_month_observations')
    # main()
    # download_from_stations()
    # download_all_data(multiprocess=True)
