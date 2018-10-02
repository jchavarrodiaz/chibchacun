# -*- coding: utf-8 -*-
import datetime
import os
import sys
import urllib2
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterstats import point_query, zonal_stats
from shapely.geometry import Point
from rasterio.errors import RasterioIOError

from config_utils import get_pars_from_ini
import sqlite3
from constants import dt_pt_ranges

# TODO: Agregar los estadisticos de la precipitacion para cada zona en la base de datos de los pronosticos

exec_prefix = sys.exec_prefix
gdal_data = '{}/share/gdal/'.format(exec_prefix)
os.environ['GDAL_DATA'] = gdal_data

dt_extents = get_pars_from_ini('../config/zones.ini')
dt_config = get_pars_from_ini('../config/config.ini')
dt_colors = get_pars_from_ini('../config/plots.ini')
dt_paths = dt_config['Paths']

# path_results = dt_paths['path_results']
path_goes = dt_paths['path_goes']
path_gis = dt_paths['path_gis']
path_eval = dt_paths['path_eval']

path_raster = '../rasters/GOES13_v1'
# path_results = 'http://172.16.1.237/almacen/externo/estaciones/interpolacion'
path_results = '../results/estaciones'

dt_months = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
             7: 'Jun', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}

path_forecast = '/home/andres/M/OF_SERVICIO_DE_PRONOSTICO_Y_ALERTAS/Compartida/' \
                '2.Análisis_pronóstico_del_tiempo/2.2_Descargas_diarias/MET_EDITOR'


def eval_raster(df_data, raster_file, min_prec=1., factor=1.):
    """
    Evaluates a raster based on point information.


    :param df_data: DataFrame with point information.
    :type df_data: pd.DataFrame
    :param raster_file: path/filename for raster image.
    :type raster_file: str
    :param min_prec: Minimum value for evaluating precipitation.
    :type min_prec: float
    :param factor: Factor for multiplying the raster.
    :type factor: float
    :return: No Rain Success, Rain Success and Mean Absolute Error (MAE)
    :rtype: dict
    """
    geometry = [Point(xy) for xy in zip(df_data['Longitude'], df_data['Latitude'])]
    df = df_data.drop(['Longitude', 'Latitude'], axis=1)
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    results = point_query(gdf, raster_file, property_name='Goes', geojson_out=True)
    gdf_stats = gpd.GeoDataFrame.from_features(results)
    gdf_stats.loc[gdf_stats['Goes'] < min_prec, 'Goes'] = 0
    gdf_stats['Goes'] = gdf_stats['Goes'] * factor
    gdf_stats.loc[gdf_stats['Value'] < min_prec, 'Value'] = 0

    idx_norain = gdf_stats[(gdf_stats['Value'] == 0) & (gdf_stats['Goes'] == 0)].index
    idx_rain = gdf_stats[(gdf_stats['Value'] > 0) & (gdf_stats['Goes'] > 0)].index

    try:
        succ_norain = len(idx_norain) / float(len(gdf_stats[(gdf_stats['Value'] == 0)]))

    except ZeroDivisionError:
        succ_norain = None

    try:
        succ_rain = len(idx_rain) / float(len(gdf_stats[(gdf_stats['Value'] > 0)]))

    except ZeroDivisionError:
        succ_rain = None

    sr_mae = gdf_stats.loc[idx_rain, 'Value'] - gdf_stats.loc[idx_rain, 'Goes']
    mae = sr_mae.abs().mean()

    return {'No Rain Success': succ_norain, 'Rain Success': succ_rain, 'Rain MAE': mae}


def eval_goes_monthly(year=None, month=None, zone='Bogota'):

    if (year is None) or (month is None):
        eval_month = pd.to_datetime(datetime.datetime.now().date().replace(day=1))

    else:
        eval_month = pd.to_datetime(datetime.date(year=year, month=month, day=1))

    year = eval_month.year
    month = eval_month.month
    dates_month = pd.date_range(start=eval_month, periods=eval_month.daysinmonth, name='Date')
    df_eval = pd.DataFrame(index=dates_month, columns=['Coverage', 'No Rain Success', 'Rain Success', 'Rain MAE'])
    zone_abb = zone[:3].upper()

    for date_eval in dates_month:
        day = date_eval.day
        path_inter = '{}/pt/csv/{:04}/{:02}/{:02}/1D'.format(path_results, year, month, day)
        file_name_inter = 'P_IDW_{0}_{1:04}{2:02}{3:02}0700.csv'.format(zone_abb, year, month, day)
        inter_file = '{}/{}'.format(path_inter, file_name_inter)
        goes_file = '{}/P24_{:04}{:02}{:02}.tif'.format(path_goes, year, month, day)

        try:
            df_inter = pd.read_csv(inter_file)
            result_eval = eval_raster(df_data=df_inter, raster_file=goes_file)
            result_eval['Coverage'] = calc_coverage(raster_file=goes_file, zone=zone)
            df_eval.loc[date_eval] = pd.Series(result_eval)

        except IOError, e:
            print(e)

    eval_name = '{0}/EVAL_{1}_{2:04}{3:02}.xlsx'.format(path_eval, zone_abb, year, month)
    df_eval.to_excel(eval_name, '{:04}{:02}'.format(year, month))


def coverage_rain(x):
    return np.ma.count(x[x >= 1.])


def calc_coverage(raster_file, zone='Bogota'):
    """
    Estimates rain coverage for a zone.

    Counts how many cells have rain based on an input zone shape
    and a raster file.
    :param raster_file: Precipitation file.
    :param zone: Zone to be analysed.
    :return: Rain coverage ratio.
    """
    shp_zone = dt_extents[zone]['shape']
    shp_filename = '{}/{}'.format(path_gis, shp_zone)
    results = zonal_stats(shp_filename, raster_file, all_touched=True, add_stats={'coverage': coverage_rain})
    coverage = [i['coverage'] for i in results]
    total = [i['count'] for i in results]
    return sum(coverage) / float(sum(total))


def precipitation_class(x):
    """
    Classifies PT values according to intervals defined in dt_pt_ranges.
    :param x: PT Matrix
    :return: Dictionary with counting of PT cells ranges.
    """

    dt_results = {}

    for pt_class in dt_pt_ranges:
        lim_inf = dt_pt_ranges[pt_class]['Lim_Inf']
        lim_sup = dt_pt_ranges[pt_class]['Lim_Sup']
        dt_results[pt_class] = np.ma.count(x[(lim_inf <= x) & (x < lim_sup)])

    return dt_results


def calc_zonal_stats(shp_input, rasterfile, stats=None):

    if stats is None:
        stats = ['count', 'max', 'min', 'mean', 'median', 'range']

    # results = zonal_stats(shp_input, rasterfile, stats=stats, all_touched=True, geojson_out=True)
    results = zonal_stats(shp_input, rasterfile,
                          stats=stats,
                          add_stats={'PT_Class': precipitation_class},
                          all_touched=True,
                          geojson_out=True)

    gdf_stats = gpd.GeoDataFrame.from_features(results)

    for i in gdf_stats.index:
        dt_class = gdf_stats.loc[i, 'PT_Class']

        if max(dt_class.values()) == 0.:
            gdf_stats.loc[i, 'Class'] = 'Seco'  # Si no se presentan lluvias, se pone cero.

        else:
            gdf_stats.loc[i, 'Class'] = max(dt_class, key=dt_class.get)

    return gdf_stats


def read_forecast_db():
    db_path = '../db/pronosticos_bogota.db'
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sql = """
    SELECT *
    FROM pronosticos2
    WHERE Fecha like '2018%'
    AND Meterologo != '7'
    ORDER BY Fecha
    """

    cur.execute(sql)
    results = cur.fetchall()
    cols = [i[0] for i in cur.description]
    df_results = pd.DataFrame(results, columns=cols)
    df_results.replace(u'Ma\xf1ana', 'Manana', inplace=True)

    df_target = df_results[['Fecha', 'Jornada', 'Zona', 'Codigo_PT']].copy()
    df_target['Fecha'] = pd.to_datetime(df_target['Fecha'], format='%Y-%m-%d %H:%M')
    df_target.loc[df_target['Jornada'] == 'Madrugada', 'Fecha'] = df_target['Fecha'] + pd.DateOffset(days=1)
    df_target['Clase'] = ''

    for class_fore in dt_pt_ranges:
        codes_pt = dt_pt_ranges[class_fore]['Codigos']
        df_target.loc[df_target['Codigo_PT'].isin(codes_pt), 'Clase'] = class_fore

    df_target.sort_values(['Fecha', 'Jornada', 'Zona'], inplace=True)
    df_target['Fecha'] = df_target['Fecha'].dt.date
    df_target.drop_duplicates(['Fecha', 'Jornada', 'Zona'], inplace=True)
    df_target.set_index(['Fecha', 'Jornada', 'Zona'], inplace=True)
    df_target.index.names = ['Fecha_PT', 'Jornada', 'Zona']
    df_target.drop('Codigo_PT', axis=1, inplace=True)
    df_target.columns = ['Clase']

    return df_target


def eval_idiger():
    shp_input = '../gis/ZonasIDIGER.shp'
    zones = sorted(list(gpd.read_file(shp_input)['Cod_Zona']))
    col_results = ['Jornada'] + zones
    stats = ['median']
    dt_results = {i: pd.DataFrame(columns=col_results) for i in stats}
    dt_estimator_class = {}
    df_pt_classes = pd.DataFrame(columns=col_results)
    df_pt_class = pd.DataFrame(columns=col_results)

    spot_hours = {'0500_': 'Noche', '1100_': 'Madrugada', '1700_': 'Manana', '2300_': 'Tarde'}
    dates = pd.date_range('2018-08-02', '2018-10-01', freq='D')

    for date_data in dates:
        path_raster_files = '{}/pt/tif/{:%Y/%m/%d}/06H'.format(path_results, date_data)
        rasterfiles = {
            i: spot_hours[i[19:24]]
            for i in os.listdir(path_raster_files)
            if (i[19:24] in spot_hours) and (i[-4:] == '.tif')
        }

        for rasterfile in sorted(rasterfiles):
            print(rasterfile)
            fore_time = rasterfiles[rasterfile]
            rasterfilename = '{}/{}'.format(path_raster_files, rasterfile)
            gdf_stats = calc_zonal_stats(shp_input, rasterfilename, stats)
            gdf_stats.set_index('Cod_Zona', inplace=True)
            sr_classes = gdf_stats['PT_Class']
            sr_classes.loc['Jornada'] = fore_time
            sr_classes.name = rasterfile
            df_pt_classes = df_pt_classes.append(sr_classes)
            sr_class = gdf_stats['Class']
            sr_class.name = rasterfile
            sr_class.loc['Jornada'] = fore_time
            df_pt_class = df_pt_class.append(sr_class)

            for stat in stats:
                sr_stat = gdf_stats[stat]
                sr_stat.loc['Jornada'] = fore_time
                sr_stat.name = rasterfile
                dt_results[stat] = dt_results[stat].append(sr_stat)

    dt_stats = {}

    for stat in stats:
        df_stat = dt_results[stat]
        df_stat['Fecha_PT'] = pd.to_datetime(dt_results[stat].index.str[11:23], format='%Y%m%d%H%M')
        df_stat.set_index('Fecha_PT', inplace=True)
        df_stat.fillna(0., inplace=True)
        df_estimator_class = pd.DataFrame(index=df_stat.index, columns=df_stat.columns)

        for class_fore in dt_pt_ranges:
            lim_low = dt_pt_ranges[class_fore]['Lim_Inf']
            lim_upp = dt_pt_ranges[class_fore]['Lim_Sup']
            df_estimator_class[(df_stat > lim_low) & (df_stat <= lim_upp)] = class_fore

        df_estimator_class['Jornada'] = df_stat['Jornada']
        df_estimator_class.reset_index(inplace=True)
        df_estimator_class.set_index(['Fecha_PT', 'Jornada'], inplace=True)
        dt_estimator_class[stat] = df_estimator_class.copy()
        sr_estimator_us = df_estimator_class.unstack(level=[0, 1]).sort_index()
        df_estimator_us = pd.DataFrame(sr_estimator_us)
        df_estimator_us.reset_index(inplace=True)
        df_estimator_us.columns = ['Zona', 'Fecha_PT', 'Jornada', 'Clase']
        df_estimator_us['Fecha_PT'] = df_estimator_us['Fecha_PT'] - pd.DateOffset(hours=11)
        df_estimator_us['Fecha_PT'] = df_estimator_us['Fecha_PT'].dt.date
        df_estimator_us.set_index(['Fecha_PT', 'Jornada', 'Zona'], inplace=True)
        df_estimator_us.sort_index(level=[0, 1, 2], inplace=True)

        df_forecast = read_forecast_db()
        idx_intersection = df_estimator_us.index.intersection(df_forecast.index)

        df_eval = pd.DataFrame(index=idx_intersection, columns=['Observado', 'Pronostico'], data='')
        df_eval['Observado'] = df_estimator_us.loc[idx_intersection, 'Clase']
        df_eval['Pronostico'] = df_forecast.loc[idx_intersection, 'Clase']
        df_eval['Acierto'] = df_eval['Observado'] == df_eval['Pronostico']
        dt_stats[stat] = df_eval
        # print(df_eval)

    df_pt_classes['Fecha_PT'] = pd.to_datetime(df_pt_classes.index.str[11:23], format='%Y%m%d%H%M')
    df_pt_classes.set_index('Fecha_PT', inplace=True)

    df_pt_class['Fecha_PT'] = pd.to_datetime(df_pt_class.index.str[11:23], format='%Y%m%d%H%M')
    df_pt_class.set_index('Fecha_PT', inplace=True)

    df_pt_class.reset_index(inplace=True)
    df_pt_class.set_index(['Fecha_PT', 'Jornada'], inplace=True)
    sr_pt_class_us = df_pt_class.unstack(level=[0, 1]).sort_index()
    df_pt_class_us = pd.DataFrame(sr_pt_class_us)
    df_pt_class_us.reset_index(inplace=True)
    df_pt_class_us.columns = ['Zona', 'Fecha_PT', 'Jornada', 'Clase']
    df_pt_class_us['Fecha_PT'] = df_pt_class_us['Fecha_PT'] - pd.DateOffset(hours=11)
    df_pt_class_us['Fecha_PT'] = df_pt_class_us['Fecha_PT'].dt.date
    df_pt_class_us.set_index(['Fecha_PT', 'Jornada', 'Zona'], inplace=True)
    df_pt_class_us.sort_index(level=[0, 1, 2], inplace=True)
    
    df_forecast = read_forecast_db()
    idx_intersection = df_pt_class_us.index.intersection(df_forecast.index)

    df_eval_pt = pd.DataFrame(index=idx_intersection, columns=['Observado', 'Pronostico'], data='')
    df_eval_pt['Observado'] = df_pt_class_us.loc[idx_intersection, 'Clase']
    df_eval_pt['Pronostico'] = df_forecast.loc[idx_intersection, 'Clase']
    df_eval_pt['Acierto'] = df_eval_pt['Observado'] == df_eval_pt['Pronostico']

    xls_output = pd.ExcelWriter('../results/idiger_stats.xlsx')
    df_pt_classes.to_excel(xls_output, 'Classes')
    df_pt_class.to_excel(xls_output, 'Class')
    df_eval_pt.to_excel(xls_output, 'Eval', merge_cells=False)
    [dt_results[i].sort_index().to_excel(xls_output, i) for i in stats]
    [dt_estimator_class[i].sort_index().to_excel(xls_output, '{}_Class'.format(i)) for i in stats]
    [dt_stats[i].to_excel(xls_output, '{}_Eval'.format(i), merge_cells=False) for i in stats]
    xls_output.save()


def rain_daily_eval(eval_date, min_precs=None, products=None):
    """

    :param eval_date:
    :type eval_date: pd.Timestamp
    :param min_precs:
    :type min_precs: float
    :param products:
    :type products: dict
    :return:
    :rtype: pd.DataFrame
    """

    if min_precs is None:
        min_precs = [1., 5., 10., 20., 30., 40., 50., 70., 100.]

    elif type(min_precs) is float:
        min_precs = [min_precs]

    if products is None:
        products = {'TRMM': .1, 'GOES13_v1': 1., 'GOES13_v2': 1.}

    elif type(products) is str:
        products = {products: 1.}

    year = eval_date.year
    month = eval_date.month
    day = eval_date.day

    path_date = '{:04}{:02}'.format(year, month)
    name_date = '{:04}{:02}{:02}'.format(year, month, day)

    try:
        df_data = pd.read_csv('{0}/pt/csv/{1:04}/{2:02}/{3:02}/1D/'
                              'P_IDW_COL_{1:04}{2:02}{3:02}0700.csv'.format(path_results, year, month, day))

    except urllib2.HTTPError:

        try:
            df_data = pd.read_csv('{0}/pt/csv/{1:04}/{2:02}/{3:02}/'
                                  'inter_pt_colombia_0700_1D.csv'.format(path_results, year, month, day))

        except urllib2.HTTPError:
            return None

    idx_metrics = ['SR', 'NR', 'MAE']
    idx_results = pd.MultiIndex.from_product((idx_metrics, products), names=['Metrica', 'Producto'])
    df_results = pd.DataFrame(columns=idx_results, index=pd.Index(min_precs, name='Precipitation'))

    for product in products:
        path_product = '{}/{}/{}'.format(path_raster, product, path_date)
        list_files = os.listdir(path_product)
        match_files = [i for i in list_files if (name_date in i) and ('.tif' in i)]
        factor = products[product]

        if len(match_files) > 0:
            raster_file = match_files[0]
            raster_fullpath = '{}/{}'.format(path_product, raster_file)

            for min_prec in min_precs:
                results_prod = eval_raster(df_data, raster_fullpath, min_prec, factor=factor)
                df_results['SR', product].loc[min_prec] = results_prod['Rain Success']
                df_results['NR', product].loc[min_prec] = results_prod['No Rain Success']
                df_results['MAE', product].loc[min_prec] = results_prod['Rain MAE']

    return df_results


def core_multiple_eval_rain(eval_date, products=None):
    print(eval_date)
    df_eval = rain_daily_eval(eval_date, products=products)
    sheet_name = '{:04}{:02}{:02}'.format(eval_date.year, eval_date.month, eval_date.day)
    return {sheet_name: df_eval}


def multiple_eval_rain(multiprocessing=False):
    xls_output = pd.ExcelWriter('../results/evaluation_rain_goes.xlsx')
    eval_dates = pd.date_range('2017-11-01', '2017-11-30')

    if multiprocessing:
        pool = Pool()
        results = pool.map(core_multiple_eval_rain, eval_dates)
        pool.close()

    else:
        results = map(core_multiple_eval_rain, eval_dates)

    results = {i.keys()[0]: i.values()[0] for i in results if i.values()[0] is not None}

    for result in sorted(results):

        if results[result] is not None:
            results[result].to_excel(xls_output, result, merge_cells=False)

    xls_output.save()


def eval_any_date():
    date_eval = pd.to_datetime('2017-11-01')
    print(core_multiple_eval_rain(date_eval, {'TRMM': .1}))


def eval_forecast():
    dates_eval = pd.date_range('2017-11-01', '2017-11-30')
    horizons_eval = ['1D', '2D']
    idx_results = pd.MultiIndex.from_product([dates_eval, horizons_eval], names=['Fecha', 'Horizonte'])
    col_results = ['SR', 'NR', 'MAE']
    df_results = pd.DataFrame(index=idx_results, columns=col_results)

    for date_eval in dates_eval:
        print(date_eval)

        year = date_eval.year
        month = date_eval.month
        month_str = dt_months[month]
        day = date_eval.day

        path_month = '{}/{}'.format(path_forecast, month_str)
        list_folders = [i for i in os.listdir(path_month) if os.path.isdir('{}/{}'.format(path_month, i))]
        str_fore_09 = '{:02}-09'.format(day)
        str_fore_15 = '{:02}-15'.format(day)
        str_fore_16 = '{:02}-16'.format(day)

        list_days = [i for i in list_folders if (i in str_fore_09) or (i in str_fore_15) or (i in str_fore_16)]

        if len(list_days) == 3:
            list_days.remove(str_fore_16)

        for day_eval in list_days:

            if day_eval != str_fore_09:
                date_eval = date_eval + pd.DateOffset(days=1)
                year = date_eval.year
                month = date_eval.month
                day = date_eval.day
                horizon = '2D'

            else:
                horizon = '1D'

            try:
                df_data = pd.read_csv('{0}/pt/csv/{1:04}/{2:02}/{3:02}/1D/'
                                      'P_IDW_COL_{1:04}{2:02}{3:02}0700.csv'.format(path_results, year, month, day))

            except urllib2.HTTPError:

                try:
                    df_data = pd.read_csv('{0}/pt/csv/{1:04}/{2:02}/{3:02}/'
                                          'inter_pt_colombia_0700_1D.csv'.format(path_results, year, month, day))

                except urllib2.HTTPError:
                    print("Archivo de estaciones no encontrado.")
                    continue

            raster_filename = '{}/{}/raster_24.tif'.format(path_month, day_eval)

            try:
                results_fore = eval_raster(df_data, raster_filename)
                df_results.loc[(date_eval, horizon), 'SR'] = results_fore['Rain Success']
                df_results.loc[(date_eval, horizon), 'NR'] = results_fore['No Rain Success']
                df_results.loc[(date_eval, horizon), 'MAE'] = results_fore['Rain MAE']

            except RasterioIOError:
                print("Raster No encontrado. {}".format(raster_filename))
                continue

    df_results.to_excel('../results/evaluation_rain_forecast.xlsx', 'IDEAM', merge_cells=False)


if __name__ == '__main__':
    # eval_goes_monthly(year=2017, month=11, zone='Colombia')
    # eval_goes_monthly()
    # multiple_eval_rain(True)
    # eval_any_date()
    # eval_forecast()
    eval_idiger()
    # read_forecast_db()
    pass
