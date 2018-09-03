# -*- coding: utf-8 -*-
from gdal_utils import calculate_pixel_size, array2raster, vector2raster
# from inter_utils import simple_idw, date_start_end, dt_sensors_names, dt_extents, fetch_data, dt_folders, dt_paths
import matplotlib
matplotlib.use('agg')
from config_utils import set_paths, build_product_name
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time

from inter_utils import *
from filter_data import filter_sta as fd


# TODO: Resolver que hacer cuando todos los elementos a interpolar son cero. Ver 2018-08-03 05:00
path_results = dt_paths['path_results']


def interpolate_pt(backward_period='1D', zone='Bogota', max_gaps=.15, sensor='0240', current_time=None):
    """


    :param backward_period:
    :param zone:
    :param max_gaps:
    :param sensor:
    :param current_time:
    :return:
    """
    # path_products = '{}/pt'.format(path_results)
    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)

    path_products = '{}/{}'.format(path_results, dt_folders[sensor])
    set_paths(path=path_products, current_time=end_time, periods=[backward_period])

    res_x, res_y = dt_extents[zone]['res_x'], dt_extents[zone]['res_y']
    x_max, x_min = dt_extents[zone]['x_max'], dt_extents[zone]['x_min']
    y_max, y_min = dt_extents[zone]['y_max'], dt_extents[zone]['y_min']
    nx = int(abs(x_max - x_min) / res_x) + 1
    ny = int(abs(y_max - y_min) / res_y) + 1

    pixel_w, pixel_h = calculate_pixel_size(x_max, x_min, y_max, y_min, nx, ny, loc=zone)

    xi = np.linspace(x_min, x_max, nx)
    yi = np.linspace(y_max, y_min, ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    basename = build_product_name(end_time, "PT", "STA", zone[:3].upper(), backward_period, True)
    df = fetch_data(zone=zone, backward_period=backward_period, sensor=sensor, current_time=current_time)

    if df is not None:
        df.to_csv('{0}/csv/{1}.csv'.format(path_products, basename))
        df_latest = df.copy()
        df_latest.to_csv('{0}/csv/pt/{1}.csv'.format(path_products.replace('pt', 'latest'), basename.split('/')[-1]))
        df_xls = df.copy()
        df_xls.drop(['Count', 'From', 'To', 'Freq', 'Total', 'AH'], axis=1, inplace=True)
        df_xls.to_excel('{0}/xls/{1}.xlsx'.format(path_products, basename), sheet_name=backward_period, merge_cells=False)

        if backward_period == '1D':
            pre_filter = df[(df['Gaps'] > 100 * (1. - max_gaps)) & (df['Gaps'] <= 100)]
            sel_stations = fd(data=pre_filter, zone=zone, sensor=sensor, current_time=current_time)

        else:
            sel_stations = df[(df['Gaps'] > 100 * (1. - max_gaps)) & (df['Gaps'] <= 100)]

        x = df.loc[sel_stations.index, 'lng'].values
        y = df.loc[sel_stations.index, 'lat'].values
        z = df.loc[sel_stations.index, 'Value'].values

        print(time.strftime("%H:%M:%S"))
        # Calculate IDW
        grid1 = simple_idw(x, y, z, xi, yi, power=4)
        grid1[grid1 <= 0] = -32768
        grid1 = grid1.reshape((ny, nx))

        metadata = {'Periodo': '{}'.format(backward_period),
                    'Registro_Inicio': '{:%Y-%m-%d %H:%M}'.format(start_time),
                    'Registro_Fin': '{:%Y-%m-%d %H:%M}'.format(end_time),
                    'Registro_Total': '{}'.format(len(sel_stations)),
                    'Region': zone,
                    'Sensor': sensor,
                    'Metodo': 'IDW'}

        # Export TIFF
        # tif_name = '{0}/tif/pt/{1}.tif'.format(path_products.replace('pt', 'latest'), basename.split('/')[-1])
        tif_name = '{0}/tif/{1}.tif'.format(path_products, basename)

        geo_trans_clip = [x_min - pixel_w / 2., pixel_w, 0, y_max + pixel_h / 2., 0, -pixel_h]
        array2raster(tif_name, geo_trans_clip, grid1, -32768, 4326, metadata)

        # # Export JPEG
        plot(x=x, y=y, z=z, grid=grid1, pixel_w=pixel_w, pixel_h=pixel_h, zone=zone)
        plt.title(u'Campo de {} (IDW) - {:%Y/%m/%d %H:00} - {}'.format(dt_sensors_names[sensor], end_time, backward_period))

        fig_name = '{0}/png/{1}.png'.format(path_products, basename)

        plt.savefig(fig_name, dpi=600)
        plt.close()

        print(time.strftime("%H:%M:%S"))


def plot(x, y, z, grid, pixel_w, pixel_h, zone='Bogota'):
    x_max, x_min = dt_extents[zone]['x_max'], dt_extents[zone]['x_min']
    y_max, y_min = dt_extents[zone]['y_max'], dt_extents[zone]['y_min']

    # masks the matrix
    mask = vector2raster(shapefile='../gis/{}.shp'.format(zone), pixel_w=pixel_w,
                         pixel_h=pixel_h, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

    # colors = [(255, 255, 255),
    #           (0, 255, 15),
    #           (0, 219, 15),
    #           (0, 146, 255),
    #           (0, 255, 255),
    #           (0, 255, 0),
    #           (0, 219, 0),
    #           (0, 146, 0),
    #           (255, 255, 0),
    #           (255, 146, 0),
    #           (255, 109, 0),
    #           (255, 0, 0),
    #           (255, 36, 0),
    #           (182, 0, 0)]
    #
    # n_bins = 14
    # cm = LinearSegmentedColormap.from_list('cm_smartmet_1D', colors=colors, N=n_bins)

    masked_array = np.ma.MaskedArray(grid, mask=np.logical_or(mask == -32768, np.logical_not(mask)))

    plt.figure()
    plt.imshow(masked_array,
               cmap='terrain',
               extent=(x_min - pixel_w / 2., x_max + pixel_w / 2., y_min - pixel_h / 2., y_max + pixel_h / 2.),
               alpha=.5)
    plt.colorbar()
    plt.scatter(x, y, c=z, s=0.1, alpha=.75)


def main():
    times = ['2018-07-30 10:35', '2018-07-31 10:35', '2018-08-01 10:35', '2018-08-02 10:35', '2018-08-03 10:35', '2018-08-04 10:35',
             '2018-08-05 10:35', '2018-08-06 10:35', '2018-08-07 10:35', '2018-08-08 10:35', '2018-08-09 10:35', '2018-08-10 10:35',
             '2018-08-11 10:35', '2018-08-12 10:35', '2018-08-13 10:35', '2018-08-14 10:35', '2018-08-15 10:35', '2018-08-16 10:35',
             '2018-08-17 10:35', '2018-08-18 10:35', '2018-08-19 10:35', '2018-08-20 10:35', '2018-08-21 10:35', '2018-08-22 10:35',
             '2018-08-23 10:35', '2018-08-24 10:35', '2018-08-25 10:35', '2018-08-26 10:35', '2018-08-27 10:35', '2018-08-28 10:35',
             '2018-08-29 10:35', '2018-08-30 10:35']
    for str_time in times:
        current_time = pd.to_datetime(str_time)
        interpolate_pt(backward_period='1D', current_time=current_time, zone='Bogota')


if __name__ == '__main__':
    main()
