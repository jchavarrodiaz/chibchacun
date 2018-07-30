# -*- coding: utf-8 -*-

import sys
from db_utils import get_config_params
import cx_Oracle as cxO
import pandas as pd
"""
Ejemplo de una consulta a oracle con agrupamiento de informacion 

SELECT CSMT_ESTACION_ID,
SUM(CSMT_PREC) data_value,
COUNT(*) count_values,
MIN(CSMT_FECHA) date_start,
MAX(CSMT_FECHA) date_end
FROM HIDROMET.SHMT_CUASIRREAL_METEOROLOGIA
WHERE to_char(CSMT_FECHA, 'yyyy-mm-dd') <= '2017-09-25'
AND to_char(CSMT_FECHA, 'yyyy-mm-dd') > '2017-09-17'
AND CSMT_PREC is not null
GROUP BY CSMT_ESTACION_ID
ORDER BY CSMT_ESTACION_ID
"""

reload(sys)
sys.setdefaultencoding('utf-8')

dt_sensors_oracle = ['CSMT_PREC', 'CSMT_TMP_MED', 'CSMT_TMP_MAX', 'CSMT_TMP_MIN', 'CSHD_NV18', 'CSHD_NV06DS']


def get_by_query_oracle(conn, sql, header=False):
    """
    Query database by sql sentence.
    :param conn:
    :type conn: cxO.Connection
    :param sql:
    :type sql: str
    :param header:
    :type header: bool
    :return:
    """
    try:
        curs = conn.cursor()
        curs.execute(sql)
        results = [i for i in curs]

        if header:
            return results, [i[0] for i in curs.description]

        else:
            return results

    except Exception, e:
        print(e)


def get_from_oracle_sql():

    params = get_config_params('../config/database.ini', section='oracle_general')
    parusr = get_config_params('../config/database.ini', section='oracle_redes')
    dsn_tns = cxO.makedsn(**params)
    conn = cxO.connect(parusr['user'], parusr['password'], dsn_tns)

    sql = """SELECT *
    FROM AGREGADOS_MENSUALES
    WHERE ESTACION LIKE '2401%'
    AND VARIABLE LIKE '%PRECIP%TOTAL%'
    """

    results = get_by_query_oracle(conn, sql)
    name_cols = ['Station', 'Date', 'Timestamp', 'Flags', 'Observations', 'Variable', 'Value']

    if len(results) > 0:
        df_results = pd.DataFrame(results)
        df_results.columns = name_cols
        # df_results.set_index('Date', inplace=True)

        df_results.to_pickle('../results/oracle/redes_2401.pkl')


def sql_catalog_oracle(table):
    """
    Returns SQL sentence in order to get catalog for the input table.
    :param table:
    :return:
    """

    if table == 'SHMT_CUASIRREAL_HIDROLOGIA':
        id_station = 'CSHD_ESTACION_ID'

    else:
        id_station = 'CSMT_ESTACION_ID'

    sql = """SELECT DISTINCT(SIOPERAN.SIOV_ESTACIONES.ID_ES), SIOPERAN.SIOV_ESTACIONES.*
    FROM SIOPERAN.SIOV_ESTACIONES, HIDROMET.{0}
    WHERE id_es = HIDROMET.{0}.{1}
    ORDER BY SIOPERAN.SIOV_ESTACIONES.ID_ES
    """.format(table, id_station)

    return sql


def sql_summary_sensor(table, sensor):
    """
    Returns SQL sentence in order to get stations that have measured for a given sensor.
    :param table:
    :type table: str
    :param sensor:
    :type sensor: str
    :return: str
    """

    if table == 'SHMT_CUASIRREAL_HIDROLOGIA':
        id_station = 'CSHD_ESTACION_ID'
    else:
        id_station = 'CSMT_ESTACION_ID'

    sql = """
    SELECT DISTINCT({0})
    FROM {1}
    WHERE {2} IS NOT NULL
    ORDER BY {0}
    """.format(id_station, table, sensor)

    return sql


def summary_sensors_stations():
    """
    Makes a summary of stations and sensors in order to identify what sensors are
    recording in each station.

    :param table_data:
    :param catalog:
    :type table_data: str
    :type catalog: str
    :return:
    """
    params = get_config_params('../config/database.ini', section='oracle_general')
    parusr = get_config_params('../config/database.ini', section='oracle_sshm')
    dsn_tns = cxO.makedsn(**params)
    conn = cxO.connect(parusr['user'], parusr['password'], dsn_tns)

    tables = {'HYDRO': 'SHMT_CUASIRREAL_HIDROLOGIA', 'METEO': 'SHMT_CUASIRREAL_METEOROLOGIA'}
    # cols_catalog = ['ID_ES', 'COD_CATALOGO_ES', 'NOMBRE_ES', 'NOMBRE_FGDA',
    #                 'GRADOS_LATITUD', 'MINUTOS_LATITUD', 'SEGUNDOS_LATITUD', 'DIRECCION_LATITUD',
    #                 'GRADOS_LONGITUD', 'MINUTOS_LONGITUD', 'SEGUNDOS_LONGITUD', 'DIRECCION_LONGITUD']

    xls_summary = pd.ExcelWriter('../results/sshm_summary.xlsx')

    for table in tables:
        sql_catalog = sql_catalog_oracle(table=tables[table])
        results, cols_catalog = get_by_query_oracle(conn=conn, sql=sql_catalog, header=True)

        df_catalog = pd.DataFrame(results)
        df_catalog.drop(0, axis=1, inplace=True)
        df_catalog.columns = cols_catalog[1:]
        df_catalog.set_index('ID_ES', inplace=True)
        df_catalog.sort_index(inplace=True)
        df_catalog.to_excel(xls_summary, 'CAT_{}'.format(table))

        df_summary = pd.DataFrame(index=df_catalog.index, columns=dt_sensors_oracle)

        for sensor in dt_sensors_oracle:
            sql_sensors = sql_summary_sensor(tables[table], sensor)
            results_sensor = get_by_query_oracle(conn, sql_sensors)

            if results_sensor is not None:
                list_stations = [i[0] for i in results_sensor]
                df_summary.loc[list_stations, sensor] = 1

            else:
                df_summary.drop(sensor, axis=1, inplace=True)

        df_summary.to_excel(xls_summary, table)

    xls_summary.save()


def get_data_stations():
    params = get_config_params('../config/database.ini', section='oracle_general')
    parusr = get_config_params('../config/database.ini', section='oracle_sshm')
    dsn_tns = cxO.makedsn(**params)
    conn = cxO.connect(parusr['user'], parusr['password'], dsn_tns)

    stations = [2171, 2181, 1288, 2098, 3519, 3536, 3544, 7427]
    xls_data = pd.ExcelWriter('../results/consulta.xlsx')

    for station in stations:
        sql = """
        SELECT CSMT_FECHA, CSMT_PREC
        FROM SHMT_CUASIRREAL_METEOROLOGIA
        WHERE CSMT_ESTACION_ID = {}
        ORDER BY CSMT_FECHA
        """.format(station)

        results = get_by_query_oracle(conn=conn, sql=sql, header=True)
        df_station = pd.DataFrame(results[0])
        df_station.columns = ['Fecha', 'Precipitacion']
        df_station.set_index('Fecha', inplace=True)
        df_station.to_excel(xls_data, str(station))

    xls_data.save()


if __name__ == '__main__':
    # summary_sensors_stations()
    get_data_stations()
    pass
