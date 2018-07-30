# -*- coding: utf-8 -*-

import pandas as pd
import simplekml


def gen_kml(forecast):
    # Create an instance of Kml
    kml = simplekml.Kml(open=1)

    # Create a point named "The World" attached to the KML document with its coordinate at 0 degrees latitude and longitude.
    # All the point's properties are given when it is constructed.
    single_point = kml.newpoint(name="The World", coords=[(0.0, 0.0)])

    # Create a point for each city. The points' properties are assigned after the point is created
    for city, time, lat, lon in forecast:
        pnt = kml.newpoint()
        pnt.name = city
        pnt.description = "Time corresponding to 12:00 noon, Eastern Standard Time: {0}".format(time)
        pnt.coords = [(lon, lat)]

    # Save the KML
    kml.save("T00 Point.kml")


icons = pd.ExcelFile('../data/forecast_bogota_icons.xlsx').parse(sheet_name='Iconos', index_col='Codigo')
points = pd.ExcelFile('../data/forecast_bogota_icons.xlsx').parse(sheet_name='Geometria', index_col='Codigo')

if __name__ == '__main__':
    gen_kml(forecast=pd.ExcelFile('../data/forecast_bogota_122020170700.xlsx').parse(sheet_name='Query'))
