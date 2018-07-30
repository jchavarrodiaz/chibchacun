# coding: utf-8
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
from osgeo import gdal
from matplotlib import patheffects, colors
from inter_utils import dt_sensors_names, dt_colors
import pandas as pd


plt.style.use('classic')


def plot_data_jpg():
    def set_colors_levels(dt_color):
        values = [[j / 255. for j in dt_color[i]] for i in sorted(dt_color)]
        colormap = colors.ListedColormap(values[1:-1])
        colormap.set_under(values[0])
        colormap.set_over(values[-1])
        bounds = [float(i) for i in sorted(dt_color.keys())[1:]]
        norm = colors.BoundaryNorm(bounds, colormap.N)

        return colormap, norm

    # ds_shade = gdal.Open("../data/SRTM_Col_HS_050.tif")
    # src_shade = ds_shade.GetRasterBand(1)
    # ndv_shade = src_shade.GetNoDataValue()
    # data_shade = ds_shade.ReadAsArray()
    # data_mk_shade = np.ma.masked_equal(data_shade, ndv_shade)

    # ds_inter = gdal.Open("../rasters/inter_ts_min_colombia_0700_1D_iso.tif")
    # ds_inter = gdal.Open("../rasters/inter_ts_max_colombia_0700_1D_iso.tif")
    # ds_inter = gdal.Open("../rasters/inter_ts_media_colombia_0700_1D_iso.tif")
    ds_inter = gdal.Open("../rasters/inter_pt_colombia_0700_1D_idw.tif")
    src_inter = ds_inter.GetRasterBand(1)
    ndv_inter = src_inter.GetNoDataValue()
    data_inter = ds_inter.ReadAsArray()
    data_mk_inter = np.ma.masked_equal(data_inter, ndv_inter)
    geoMatrix = ds_inter.GetGeoTransform()
    nx = data_inter.shape[1]
    ny = data_inter.shape[0]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    min_x = geoMatrix[0]
    max_x = min_x + xDist * nx
    max_y = geoMatrix[3]
    min_y = max_y + yDist * ny
    metadata = ds_inter.GetMetadata()
    sensor = metadata['Sensor']
    dt_color = dt_colors['colors_{}'.format(sensor)]
    colormap, norm = set_colors_levels(dt_color)

    size_factor = 1.6
    fig = plt.figure(figsize=[3 * size_factor, 4 * size_factor])

    # plot basemap, rivers and countries
    bmap = Basemap(llcrnrlat=min_y, urcrnrlat=max_y, llcrnrlon=min_x, urcrnrlon=max_x, epsg=4326, projection='tmerc',
                   lat_0=7.165, lon_0=-74.085, resolution='i', area_thresh=10000)

    # m.etopo().set_alpha(0.25)
    # bmap.shadedrelief().set_alpha(.25)
    bmap.bluemarble().set_alpha(.2)
    bmap.readshapefile('../gis/Departamentos', 'Departamentos', linewidth=.4)
    # m.drawcoastlines(color='k', linewidth=0.3)
    # m.drawcountries(color='k', linewidth=0.3)
    # m.drawstates(color='0.25')
    # bmap.drawrivers(color='dodgerblue', linewidth=0.2, zorder=1).set_alpha(0.5)

    # Set parallels and meridians
    bmap.drawmeridians(np.arange(-80, -65, 2), labels=[0, 0, 1, 1], linewidth=0.1, color='k', fontstyle='italic',
                       fontsize='xx-small', dashes=[2, 2])
    bmap.drawparallels(np.arange(-6, 16, 2), labels=[1, 1, 0, 0], linewidth=0.1, color='k', fontstyle='italic',
                       fontsize='xx-small', rotation=90, dashes=[2, 2])

    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(max_y, min_y, ny)

    xx, yy = np.meshgrid(x, y)

    # hillshade = bmap.imshow(data_mk_shade)
    colormesh = bmap.pcolormesh(xx, yy, data_mk_inter, latlon=True, cmap=colormap, norm=norm)
    # colormesh = bmap.contourf(xx, yy, data_mk_inter, latlon=True, cmap=colormap)  # , norm=norm)

    plt.legend(loc='lower left', ncol=2, fontsize=5, fancybox=True, frameon=True)
    plt.tight_layout()

    # Set title
    title = dt_sensors_names[sensor]
    start = pd.to_datetime(metadata['Registro_Inicio'])
    end = pd.to_datetime(metadata['Registro_Fin'])

    text = u"""Regresión a partir de estaciones y elevación.
    {:%Y-%m-%d %H:%M} - {:%Y-%m-%d %H:%M} (HLC)""".format(start, end)

    ax_title = fig.add_axes([.3, .71, .65, .2], frameon=False)
    ax_title.axis('off')
    ax_title.set_title(title, fontsize='small', weight='bold', loc='right', color='k',
                       path_effects=[patheffects.withStroke(linewidth=2., foreground="w")])
    ax_title.text(1., 1., text, fontsize='x-small', ha='right', va='top', linespacing=1.,
                  path_effects=[patheffects.withStroke(linewidth=1., foreground="w")])

    # Set colorbar
    ax_cb = fig.add_axes([0.1, 0.11, 0.4, 0.025], frameon=True)
    ax_cb.axis('off')
    # ax_cb.axes.get_xaxis().set_visible(False)
    # ax_cb.axes.get_yaxis().set_visible(False)

    cbar = bmap.colorbar(colormesh, location='bottom', pad='-100%', size='100%', ax=ax_cb, extend='both')
    # cbar.set_over('r')
    # cbar.set_under('b')
    cbar.set_label(label=ur'Temperatura [ºC]', weight='bold', fontsize='x-small',
                   path_effects=[patheffects.withStroke(linewidth=1.5, foreground="w")])
    cbar.ax.tick_params(axis='x', direction='out')
    cbar.ax.set_xticklabels(cbar.ax.xaxis.get_ticklabels(), fontsize='xx-small', weight='bold',
                            path_effects=[patheffects.withStroke(linewidth=1., foreground="w")])
    cbar.extendfrac = 0.15
    colormesh.changed()

    # Set north arrow
    im_north = plt.imread('../data/north1.png')
    ax_na = fig.add_axes([0.05, 0.85, 0.1, 0.1], anchor='NE', zorder=1)
    ax_na.imshow(im_north)
    ax_na.axis('off')

    # Set logo
    im_logo = plt.imread('../data/logo_ideam_1.png')
    ax_li = fig.add_axes([.75, .05, .15, .15], zorder=1)
    ax_li.imshow(im_logo)
    ax_li.axis('off')

    plt.savefig('../figs/prueba_{}'.format(sensor), dpi=300)
    plt.close()


plot_data_jpg()
