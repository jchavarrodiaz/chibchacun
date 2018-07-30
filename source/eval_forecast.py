# -*- coding: utf-8 -*-
import ftplib
import pandas as pd
from config_utils import make_dir


months = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
          7: 'Jun', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}

path_forecast = u'/home/andres/M/OF_SERVICIO_DE_PRONOSTICO_Y_ALERTAS/Compartida/' \
                u'2.Análisis_pronóstico_del_tiempo/2.2_Descargas_diarias/MET_EDITOR'


def download_trmm(download_date, hour=12, horizons=None):
    """
    Downloads TRMM data for a date, hour and horizons given.

    :param download_date:
    :type download_date: pd.Timestamp
    :param hour:
    :type hour: int
    :param horizons:
    :type horizons: list
    :return:
    """
    year = download_date.year
    month = download_date.month
    day = download_date.day

    folder_name = '{:04}{:02}'.format(year, month)

    path_ftp_jaxa = 'trmmopen.gsfc.nasa.gov'
    folder_jaxa = 'pub/gis/{}'.format(folder_name)
    path_download = '../rasters/TRMM/{}'.format(folder_name)
    make_dir(path_download)

    ftp = ftplib.FTP(path_ftp_jaxa)
    ftp.login()
    ftp.cwd(folder_jaxa)
    list_data = ftp.nlst()

    if horizons is None:
        horizons = ['1day', '3day', '7day']

    for horizon in horizons:
        text_horizon = '{}{:02}{}.7.{}'.format(folder_name, day, hour, horizon)
        list_horizon = [i for i in list_data if text_horizon in i]

        for file_horizon in list_horizon:
            path_file = '{}/{}'.format(path_download, file_horizon)
            ftp.retrbinary('RETR {}'.format(file_horizon), open(path_file, 'wb').write)

    ftp.quit()


def main():
    download_dates = pd.date_range('2017-11-01', '2017-12-30')

    for download_date in download_dates:
        download_trmm(download_date)


if __name__ == '__main__':
    main()
    pass
