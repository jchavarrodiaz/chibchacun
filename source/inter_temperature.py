# -*- coding: utf-8 -*-

from scipy import stats
from inter_utils import *
import matplotlib
matplotlib.use('agg')
from validation_utils import grubbs_test
from config_utils import set_paths, build_product_name
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.style.use('classic')
path_results = dt_paths['path_results']


def as_si(x, ndp=3):
    """
    Sets float numbers as scientific notation using - x 10 ^ - form.

    Use the form 1.54 x 10 ^ -5 instead of 1.54E-5
    :param x: float number.
    :param ndp: number of decimal places.
    :return:
    """
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times10^{{{e:d}}}'.format(m=m, e=int(e))


def interpolate_ts(backward_period='1D', zone='Bogota', max_gaps=.15, sensor='0068', current_time=None):
    """

    :param backward_period:
    :param zone:
    :param max_gaps:
    :param sensor:
    :param current_time:
    :return:
    """
    def plot_h_vs_t():
        """
        Scatter plot elevation vs. temperature.

        :return:
        """
        reg_name = '{0}/png/{1}_ECU.png'.format(path_products, basename)

        eq_reg = '$T = {}H + {:.2f}$\n$R^2 = {:.2f}$'.format(as_si(alpha, 2), beta, r2)

        if sr_area is not None:
            areas = sorted(sr_area.unique())
            colors = cm.rainbow(np.linspace(0, 1, len(areas)))

            for area, color in zip(areas, colors):
                plt.scatter(sr_elevation[sr_area == area], sr_temperature[sr_area == area], c=color,
                            label=dt_ah_names[area])

        else:
            plt.scatter(sr_elevation, sr_temperature, s=1.)

        plt.title(u'Regresión Elevación vs. {}'.format(dt_sensors_names[sensor]), fontsize='large')
        plt.ylabel(ur'Temperatura [$^{\circ}C$]', fontsize='medium')
        plt.xlabel(ur'Elevación [$msnm$]', fontsize='medium')
        plt.gca().set_xlim(left=0)
        plt.xticks(fontsize='small')
        plt.yticks(fontsize='small')
        legend = plt.legend(fontsize='x-small', title=u'Área Hidrográfica')
        plt.setp(legend.get_title(), fontsize='medium')
        x_plot_min = plt.axes().get_xlim()[0]
        x_plot_max = plt.axes().get_xlim()[1]
        y_plot_min = plt.axes().get_ylim()[0]
        y_plot_max = plt.axes().get_ylim()[1]
        plt.gca().set_ylim(bottom=y_plot_min)
        x_eq = x_plot_max * .2
        y_eq = y_plot_min + (y_plot_max - y_plot_min) * .1
        plt.annotate(eq_reg, xy=(x_eq, y_eq), va='center', ha='center', bbox=dict(boxstyle="round", fc="w"),
                     fontsize='medium')
        x_line = [x_plot_min, x_plot_max]
        line_plot = [alpha * h + beta for h in x_line]
        plt.plot(x_line, line_plot, '--k')
        plt.grid()
        plt.tight_layout()
        plt.savefig(reg_name, dpi=300)
        plt.close()

    folder = dt_folders[sensor]
    type_ts = folder.replace('/', '').upper()
    start_time, end_time = date_start_end(backward_period=backward_period, current_time=current_time)

    path_products = '{}/{}'.format(path_results, folder)
    set_paths(path=path_products, current_time=end_time, periods=[backward_period])

    basename = build_product_name(end_time, type_ts, "STA", zone[:3].upper(), backward_period, True)
    df = fetch_data(zone=zone, backward_period=backward_period, sensor=sensor, current_time=current_time)

    if df is not None:
        sel_stations = df[(df['Gaps'] > 100 * (1. - max_gaps)) & (df['Gaps'] <= 100)].index
        gaps_stations = df.index.symmetric_difference(sel_stations)
        df.loc[gaps_stations, 'Observation'] = "Gaps"

        outliers = True
        alpha = None
        beta = None
        r2 = None

        while outliers:
            sr_elevation = df.loc[sel_stations, 'Elevation']
            sr_temperature = df.loc[sel_stations, 'Value']

            reg_results = stats.linregress(x=sr_elevation, y=sr_temperature)
            alpha = reg_results[0]
            beta = reg_results[1]
            corr = reg_results[2]
            r2 = corr ** 2

            sr_interpolation = sr_elevation * alpha + beta
            sr_error = sr_temperature - sr_interpolation
            df_clean, df_outliers = grubbs_test(sr_error)
            sel_stations = df_clean.dropna().index
            outlier_stations = df_outliers.dropna().index
            df.loc[outlier_stations, 'Observation'] = "Outlier"

            if len(outlier_stations) > 0:
                outliers = True

            else:
                outliers = False

        df.to_csv('{0}/csv/{1}.csv'.format(path_products, basename))
        df.to_excel('{0}/xls/{1}.xlsx'.format(path_products, basename), sheet_name=backward_period, merge_cells=False)

        sr_area = df.loc[sel_stations, 'AH']
        dem_path = dt_paths['path_dems']
        dem_file = dt_extents[zone]['dem']
        dem_filename = '{}/{}'.format(dem_path, dem_file)
        raster = gdal.Open(dem_filename)
        band = raster.GetRasterBand(1)
        trans_data = raster.GetGeoTransform()
        no_data_value = band.GetNoDataValue()
        dem_data = BandReadAsArray(band)
        temperature_grid = alpha * dem_data + beta
        temperature_grid[dem_data == no_data_value] = no_data_value

        # Export TIFF
        metadata = {'Periodo': '{}'.format(backward_period),
                    'Registro_Inicio': '{:%Y-%m-%d %H:%M}'.format(start_time),
                    'Registro_Fin': '{:%Y-%m-%d %H:%M}'.format(end_time),
                    'Registro_Total': '{}'.format(len(sel_stations)),
                    'Regresion_alpha': '{}'.format(alpha),
                    'Regresion_beta': '{}'.format(beta),
                    'Regresion_R2': '{}'.format(r2),
                    'Region': zone,
                    'Sensor': sensor,
                    'Metodo': 'Isotermas'}

        tif_name = '{0}/tif/{1}.tif'.format(path_products, basename)

        array2raster(tif_name, trans_data, temperature_grid, -32768, 4326, metadata)

        # Export JPEG
        plot_h_vs_t()


def main():
    sensors = ['0069', '0070']

    for sensor in sensors:
        interpolate_ts('1D', 'Bogota', sensor=sensor)


if __name__ == '__main__':
    main()
