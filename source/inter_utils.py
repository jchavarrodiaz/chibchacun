# -*- coding: utf-8 -*-
import cx_Oracle as cxO
import numpy as np
import pandas as pd
import prestodb
import requests
from osgeo import gdal
from osgeo.gdalnumeric import BandReadAsArray

from config_utils import get_pars_from_ini, date_start_end
from conn_oracle import get_by_query_oracle
from conn_presto import get_by_query_presto, dt_sensors_postgres, dt_fields
from db_utils import get_config_params
from gdal_utils import array2raster

dt_sensors_oracle = {'0240': 'CSMT_PREC', '0068': 'CSMT_TMP_MED', '0069': 'CSMT_TMP_MAX', '0070': 'CSMT_TMP_MIN'}
dt_sensors_agg = {'0068': 'AVG', '0069': 'MAX', '0070': 'MIN', '0240': 'SUM'}
dt_sensors_abb = {'0068': 'TS', '0069': 'TS', '0070': 'TS', '0240': 'PT'}
dt_sensors_vars = {'0068': u'Temperatura',
                   '0069': u'Temperatura',
                   '0070': u'Temperatura',
                   '0240': u'Precipitación'}

dt_sensors_names = {'0068': u'Temperatura Media del aire a 2m',
                    '0069': u'Temperatura Máxima del aire a 2m',
                    '0070': u'Temperatura Mínima del aire a 2m',
                    '0240': u'Precipitación Total'}

dt_sensors_units = {'0068': u'ºC',
                    '0069': u'ºC',
                    '0070': u'ºC',
                    '0240': u'mm'}

dt_folders = {'0068': 'ts/media',
              '0069': 'ts/max',
              '0070': 'ts/min',
              '0240': 'pt'}

dt_ah_names = {1: u'Caribe', 2: u'Magdalena-Cauca', 3: u'Orinoco', 4: u'Amazonas', 5: u'Pacífico'}

dt_extents = get_pars_from_ini('../config/zones.ini')
dt_config = get_pars_from_ini('../config/config.ini')
dt_colors = get_pars_from_ini('../config/plots.ini')
dt_paths = dt_config['Paths']


def select_stations(df_catalog, zone='Bogota'):
    """
    Returns stations located in the zone defined by extents in zones.ini file.

    :param df_catalog: Station's catalog with geographic coordinates.
    :type df_catalog: DataFrame
    :param zone: The zone must be included in config/zones.ini.
    :type zone: str
    :return:
    """
    x_min = dt_extents[zone]['x_min']
    x_max = dt_extents[zone]['x_max']
    y_min = dt_extents[zone]['y_min']
    y_max = dt_extents[zone]['y_max']

    df_zone = df_catalog[(x_min <= df_catalog['lng']) & (df_catalog['lng'] <= x_max) &
                         (y_min <= df_catalog['lat']) & (df_catalog['lat'] <= y_max)]

    return df_zone


def distance_matrix(x, y, xi, yi):
    """
    Returns distance matrix.

    :param x: x coordinates vector
    :param y: y coordinates vector
    :param xi: x coordinates to interpolate
    :param yi: y coordinates to interpolate
    :return:
    """
    obs = np.vstack((x, y)).T
    interp = np.vstack((xi, yi)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)


def simple_idw(x, y, z, xi, yi, power=2):
    """
    Performs Simple IDW interpolation.

    :param x: x coordinates vector
    :param y: y coordinates vector
    :param z: values vector for interpolating
    :param xi: x coordinates to interpolate
    :param yi: y coordinates to interpolate
    :param power: distance power
    :return:
    """
    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / (dist ** power)

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi


def calc_isotherm(alpha, beta, dem_filename, isotherm_filename):
    """
    Calculates isotherm values based on a dem raster file and regression results.

    :param alpha:
    :param beta:
    :param dem_filename:
    :param isotherm_filename:
    :return:
    """
    raster = gdal.Open(dem_filename)
    band = raster.GetRasterBand(1)
    trans_data = raster.GetGeoTransform()
    no_data_value = band.GetNoDataValue()
    data = BandReadAsArray(band)
    output = alpha * data + beta
    output[data == no_data_value] = no_data_value
    array2raster(isotherm_filename, geo_transform=trans_data, array=output, no_data_value=no_data_value)


def query_sentence_presto(station, max_value, min_value=0, catalog='postgresql', schema='data_radio',
                          table='recent_data', sensor='0240', backward_period='1H', current_time=None):
    """
    Returns a SQL sentence for passing it to presto db.

    :param station: station code, it depends on database selected.
    :param max_value:
    :param min_value:
    :param catalog:
    :param schema:
    :param table:
    :param sensor:
    :param backward_period:
    :param current_time:
    :return:
    """
    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)
    str_start_time = '{:%Y-%m-%d %H:%M}'.format(start_time)
    str_end_time = '{:%Y-%m-%d %H:%M}'.format(end_time)

    id_station = dt_fields[catalog]['station']
    id_sensor = dt_fields[catalog]['sensor']
    id_value = dt_fields[catalog]['value']
    id_time = dt_fields[catalog]['time']

    if catalog == 'cassandra':
        val_station = "'{:010}'".format(station)
        val_sensor = "'{}'".format(sensor)
        id_time = "{} + interval '5' hour".format(id_time)

    else:
        val_station = station
        val_sensor = dt_sensors_postgres[sensor]

    val_agg = dt_sensors_agg[sensor]

    sql_query = """
    SELECT {0} station, {1}({2}) data_value, count(*) total_values,
    min(to_char({3},'yyyy-mm-dd hh24:mi')) datemin, max(to_char({3},'yyyy-mm-dd hh24:mi')) datemax
    FROM {4}.{5}
    WHERE {0}={6} AND {7}={8}
    AND to_char({3},'yyyy-mm-dd hh24:mi') > '{9}'
    AND to_char({3},'yyyy-mm-dd hh24:mi') <= '{10}'
    AND {2} >= {11} AND {2} <= {12}
    GROUP BY {0}
    ORDER by 2
    """.format(id_station, val_agg, id_value, id_time, schema, table, val_station, id_sensor, val_sensor,
               str_start_time, str_end_time, min_value, max_value)

    return sql_query


def query_sentence_oracle(table='SHMT_CUASIRREAL_METEOROLOGIA', sensor='0240', backward_period='1D',
                          current_time=None):
    """
    Returns a SQL sentence for pass it to oracle db.

    :param table:
    :param sensor:
    :param backward_period:
    :param current_time:
    :return:
    """
    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)

    if sensor != '0070':
        start_time = start_time - pd.DateOffset(days=1)
        end_time = end_time - pd.DateOffset(days=1)

    val_agg = dt_sensors_agg[sensor]

    sql_query = """
    SELECT CSMT_ESTACION_ID,
    {0}({1}) data_value,
    COUNT(*) count_values,
    MIN(CSMT_FECHA) date_start,
    MAX(CSMT_FECHA) date_end
    FROM HIDROMET.{2}
    WHERE to_char(CSMT_FECHA, 'yyyy-mm-dd') > '{3:%Y-%m-%d}'
    AND to_char(CSMT_FECHA, 'yyyy-mm-dd') <= '{4:%Y-%m-%d}'
    AND {1} is not null
    GROUP BY CSMT_ESTACION_ID
    ORDER BY CSMT_ESTACION_ID
    """.format(val_agg, dt_sensors_oracle[sensor], table, start_time, end_time)

    return sql_query


def get_df_data_presto(df_catalog, backward_period='1D', sensor='0240',
                       section='prestodb_postgresql', table='recent_data', current_time=None):
    """
    Returns a dataframe from a query to prestodb based on a zone catalog.

    It works for any database managed by presto.
    :param df_catalog:
    :param backward_period:
    :param sensor:
    :param section:
    :param table:
    :param current_time:
    :return:
    """
    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)
    stations = df_catalog.index
    params = get_config_params('../config/database.ini', section=section)
    catalog = params['catalog']
    schema = params['schema']
    conn = prestodb.dbapi.Connection(**params)
    name_cols = ['Station', 'Value', 'Count', 'From', 'To']
    dict_total = {}
    sensor_abb = dt_sensors_abb[sensor]

    for query_station in stations:
        max_value = df_catalog.loc[query_station, 'max {}'.format(dt_sensors_abb[sensor]).lower()]

        if sensor == '0240':
            min_value = 0

        else:
            min_value = df_catalog.loc[query_station, 'min {}'.format(dt_sensors_abb[sensor]).lower()]

        if pd.notnull(max_value):
            name = df_catalog.loc[query_station, 'nombre']
            freq = df_catalog.loc[query_station, 'frec_{}'.format(sensor_abb.lower())]
            longitude = df_catalog.loc[query_station, 'lng']
            latitude = df_catalog.loc[query_station, 'lat']
            total_data = (end_time - start_time) / pd.Timedelta('{}M'.format(freq))
            elevation = df_catalog.loc[query_station, 'elevacion']
            hydro_area = df_catalog.loc[query_station, 'ah']

            sql = query_sentence_presto(station=query_station,
                                        max_value=max_value,
                                        min_value=min_value,
                                        catalog=catalog,
                                        schema=schema,
                                        table=table,
                                        sensor=sensor,
                                        backward_period=backward_period,
                                        current_time=current_time)

            results = get_by_query_presto(conn, sql)
            conn.close()

            if len(results) > 0:
                df_results = pd.DataFrame(results)
                df_results.columns = name_cols
                df_results['Name'] = name
                df_results['Freq'] = freq
                df_results['Total'] = int(total_data)
                df_results['Gaps'] = 100. * df_results.loc[0, 'Count'] / total_data
                df_results['lng'] = longitude
                df_results['lat'] = latitude
                df_results['Elevation'] = elevation
                df_results['AH'] = hydro_area
                dict_total[query_station] = df_results

    df_total = pd.concat(dict_total)
    df_total.sort_values(['Value'], ascending=False, inplace=True)
    df_total.reset_index(inplace=True, drop=True)
    df_total.set_index(['Station'], inplace=True)
    return df_total


def get_df_data_oracle(df_catalog, backward_period='1D', sensor='0240', current_time=None):
    """
    Returns a dataframe from a query to oracle db based on a zone catalog.

    :param df_catalog:
    :param backward_period:
    :param sensor:
    :param current_time:
    :return:
    """
    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)

    if sensor != '0070':
        start_time = start_time - pd.DateOffset(days=1)
        end_time = end_time - pd.DateOffset(days=1)

    params = get_config_params('../config/database.ini', section='oracle_general')
    parusr = get_config_params('../config/database.ini', section='oracle_sshm')
    dsn_tns = cxO.makedsn(**params)
    conn = cxO.connect(parusr['user'], parusr['password'], dsn_tns)
    query_table = 'SHMT_CUASIRREAL_METEOROLOGIA'
    name_cols = ['Station', 'Value', 'Count', 'From', 'To']

    sql = query_sentence_oracle(table=query_table,
                                sensor=sensor,
                                backward_period=backward_period,
                                current_time=current_time)

    results = get_by_query_oracle(conn, sql)
    conn.close()

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        df_results.columns = name_cols
        df_results.set_index('Station', inplace=True)
        stations = df_results.index.intersection(df_catalog.index)
        df_results = df_results.loc[stations]
        df_results.loc[stations, 'Name'] = df_catalog.loc[stations, 'nombre']
        df_results.loc[stations, 'Freq'] = df_catalog.loc[stations, 'frecuencia']
        df_results['Total'] = int((end_time - start_time) / pd.Timedelta('1D'))
        df_results['Gaps'] = 100. * df_results['Count'] / df_results['Total']
        df_results.loc[stations, 'lng'] = df_catalog.loc[stations, 'lng']
        df_results.loc[stations, 'lat'] = df_catalog.loc[stations, 'lat']
        df_results.loc[stations, 'Elevation'] = df_catalog.loc[stations, 'elevacion']
        df_results.loc[stations, 'AH'] = df_catalog.loc[stations, 'ah']
        df_results.sort_values(['Value'], ascending=False, inplace=True)

        return df_results

    else:
        return None


def get_df_data_sshm(df_catalog, backward_period='1D', sensor='0240', current_time=None):
    """
    Returns a dataframe from a query to Wladimir's csv file based on a zone catalog.

    :param df_catalog:
    :param backward_period:
    :param sensor:
    :param current_time:
    :return:
    """
    dt_sensors_wlado = {'0240': 'PTPM_CON',
                        '0068': 'TSSM_CON'}

    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)

    # if sensor != '0070':
    #     start_time = start_time - pd.DateOffset(days=1)
    #     end_time = end_time - pd.DateOffset(days=1)

    url = 'http://datos.ideam.gov.co/datainmotion-datax/productos/generales/estaciones/cuasirreales/crudos/{}/estaciones.csv'.format(end_time._date_repr)
    r = requests.get(url, allow_redirects=True)
    if r.reason == 404:
        results = None
    else:
        open('../temp/sshm.csv', 'wb').write(r.content)
        df_query = pd.read_csv('../temp/sshm.csv', header=None, names=['Station', 'Variable', 'From', 'Value'])

        df_query['From'] = [pd.Timestamp(x, tz=start_time.tz) for x in df_query['From']]
        df_query['To'] = end_time

        results = df_query[(df_query['Variable'] == dt_sensors_wlado[sensor]) & (df_query['From'] >= start_time) & (df_query['From'] < end_time)].drop(labels=['Variable'], axis=1)

    df_catalog_new = df_catalog.copy()
    df_catalog_new.set_index('codigo_general', inplace=True)

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        df_results.set_index('Station', inplace=True)
        stations = df_results.index.intersection(df_catalog.set_index('codigo_general').index)
        df_results = df_results.loc[stations]
        df_results.loc[stations, 'Name'] = df_catalog_new.loc[stations, 'nombre']
        df_results.loc[stations, 'Freq'] = df_catalog_new.loc[stations, 'frecuencia']
        df_results['Count'] = [list(df_results.index).count(x) for x in df_results.index]
        df_results['Total'] = int((end_time - start_time) / pd.Timedelta('1D'))
        df_results['Gaps'] = 100. * df_results['Count'] / df_results['Total']
        df_results.loc[stations, 'lng'] = df_catalog_new.loc[stations, 'lng']
        df_results.loc[stations, 'lat'] = df_catalog_new.loc[stations, 'lat']
        df_results.loc[stations, 'Elevation'] = df_catalog_new.loc[stations, 'elevacion']
        df_results.loc[stations, 'AH'] = df_catalog_new.loc[stations, 'ah']
        df_results.sort_values(['Value'], ascending=False, inplace=True)

        df_results.index.name = 'Station'

        return df_results

    else:
        return None


def fetch_data(zone='Bogota', backward_period='1D', sensor='0240',
               current_time=None):
    """
    Collects data from all available databases based on zone and returns a dataframe.

    :param zone:
    :param backward_period:
    :param sensor:
    :param current_time:
    :return:
    """
    if backward_period[-1] == 'D':
        databases = ['cassandra', 'postgresql', 'sshm']

    else:
        databases = ['cassandra', 'postgresql']

    # xls_catalog = pd.ExcelFile('../data/catalogos_ideam.xlsx')
    xls_catalog = pd.ExcelFile('../data/catalogos/catalogos_ideam.xlsx')
    dt_data = {}

    for database in databases:
        df_catalog = xls_catalog.parse(database, index_col='codigo')
        df_catalog.drop(df_catalog.query('Falla == True').index, inplace=True)
        df_catalog_zone = select_stations(df_catalog=df_catalog, zone=zone)

        if database == 'cassandra':
            try:
                df_data = get_df_data_presto(df_catalog=df_catalog_zone,
                                             backward_period=backward_period,
                                             sensor=sensor,
                                             section='prestodb_cassandra',
                                             table='last_month_observations',
                                             # table='weather_events',
                                             current_time=current_time)

                dt_data['cassandra'] = df_data.copy()

            except Exception, e:
                print "No se pudo obtener informacion de la base {}. {}".format(database, e)
                dt_data['cassandra'] = None

        elif database == 'postgresql':
            try:
                df_data = get_df_data_presto(df_catalog=df_catalog_zone,
                                             backward_period=backward_period,
                                             sensor=sensor,
                                             section='prestodb_postgresql',
                                             table='recent_data',
                                             # table='archive_data',
                                             current_time=current_time)

                dt_data['postgresql'] = df_data.copy()

            except Exception, e:
                print "No se pudo obtener informacion de la base {}. {}".format(database, e)
                dt_data['postgresql'] = None

        elif database == 'sshm':
            try:

                df_data = get_df_data_sshm(df_catalog=df_catalog_zone,
                                           backward_period=backward_period,
                                           sensor=sensor,
                                           current_time=current_time)

                dt_data['sshm'] = df_data.copy()

            except Exception, e:
                print "No se pudo obtener informacion de la base {}. {}".format(database, e)
                dt_data['sshm'] = None

    dt_data = {i: dt_data[i] for i in dt_data if i is not None}

    if len(dt_data) > 0:
        df_total = pd.concat(dt_data, names=['Database'])

        if sensor == '0070':
            ascending = True

        else:
            ascending = False

        df_total.sort_values(['Value', 'Name'], ascending=ascending, inplace=True)

        return df_total

    else:
        return None


if __name__ == '__main__':
    pass
