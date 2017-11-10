"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

import res.globvars as globvars
import pandas as pd
import utm
import os

MIN_YEAR = 2015
USABLE_CRIMES = [
    'Mischief',
    'Theft from Vehicle' ,
    'Break and Enter Residential/Other',
    'Theft of Vehicle',
    'Break and Enter Commercial',
    'Other Theft',
    'Theft of Bicycle'
]

CRIME_FILE = globvars.RESOURCE_FILES_PATH + "crime_data.csv"
CITY_FILE = globvars.RESOURCE_FILES_PATH + "city_data.csv"


# Contains all functions to clean the raw data
class DataCleaner:
    def CleanRawData():
        DataCleaner.CleanCrimeData()

        if not os.path.isfile(CITY_FILE):
            city_data = pd.DataFrame(columns=["X", "Y", "TYPE"])
            city_data = city_data.append(DataCleaner.CleanRTSData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanParksData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanCentersData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanGardensData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanGraffitiData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanGreenestCityData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanHomelessSheltersData(), ignore_index=True)
            city_data = city_data.append(DataCleaner.CleanSchoolsData(), ignore_index=True)
            city_data = city_data.dropna(axis=0)
            city_data.to_csv(CITY_FILE, index=False)

        return CRIME_FILE, CITY_FILE


    def CleanCrimeData():
        if os.path.isfile(CRIME_FILE): return
        crimes = pd.read_csv(globvars.RAW_DATA_FILEPATH + 'crime_csv_all_years.csv',
                             sep=',', header=0, usecols=['TYPE','YEAR','HUNDRED_BLOCK','X','Y'])

        crimes = crimes[crimes.YEAR >= MIN_YEAR]
        # The X and Y values are set to zero for any crime against a person to
        #   protect their identity, which means we can't use the data
        crimes = crimes[(crimes.X != 0.0) | (crimes.Y != 0.0)]
        crimes = crimes[crimes['TYPE'].isin(USABLE_CRIMES)]

        crimes.to_csv(CRIME_FILE, index=False)


    def ConvertToXY(x, feature="NA"):
        try:
            utm_coors = utm.from_latlon(x['Lat'], x['Long'])
        except Exception as e:
            print(e)
            return

        # Vancouver is located in zone 10, the value should not change
        assert utm_coors[2] == 10
        return pd.Series({'X': utm_coors[0], 'Y': utm_coors[1], 'TYPE': feature})


    # Returns a dataframe containing the X,Y coordinates for each rapid transit station
    def CleanRTSData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "rapid_transit_stations.csv",
                          usecols=['X','Y'])
        data = data.dropna(axis=0)
        data.columns = ['Long', 'Lat']
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="RTS")
        return cleaned_data


    # Returns a dataframe containing the X,Y coordinates of each park
    def CleanParksData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "parks.csv",
                            usecols=["GoogleMapDest"], index_col=None)
        data = data.dropna(axis=0)
        latlongs = data.apply(DataCleaner.SplitLatLong, axis=1)
        cleaned_parks = latlongs.apply(DataCleaner.ConvertToXY, axis=1, feature="PARK")
        return cleaned_parks


    # Splits GoogleMapDest: '49.288336,-123.036982' into separate columns
    def SplitLatLong(x):
        coords = x['GoogleMapDest'].split(',')
        return pd.Series({'Lat': float(coords[0]), 'Long': float(coords[1])})


    # Returns a dataframe containing the X,Y coordinates
    def CleanCentersData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "community_centres.csv",
                            usecols=['LATITUDE', 'LONGITUDE'], index_col=None)
        data.columns = ['Lat', 'Long']
        data = data.dropna(axis=0)
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="COMMUNITYCENTER")
        return cleaned_data


    # Returns a dataframe containing the X,Y coordinates
    def CleanGardensData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "CommunityGardensandFoodTrees.csv",
                            usecols=['LATITUDE', 'LONGITUDE'], index_col=None)
        data.columns = ['Lat', 'Long']
        data = data.dropna(axis=0)
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="GARDEN")
        return cleaned_data


    # Returns a dataframe containing the X,Y coordinates
    def CleanGraffitiData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "graffiti.csv",
                            usecols=['X', 'Y'], index_col=None)
        data.columns = ['Long', 'Lat']
        data = data.dropna(axis=0)
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="GRAFFITI")
        return cleaned_data


    # Returns a dataframe containing the X,Y coordinates
    def CleanGreenestCityData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "greenest_city_projects.csv",
                            usecols=['LATITUDE', 'LONGITUDE'], index_col=None)
        data.columns = ['Lat', 'Long']
        data = data.dropna(axis=0)
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="CITYPROJECT")
        return cleaned_data


    # Returns a dataframe containing the X,Y coordinates
    def CleanHomelessSheltersData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "homeless_shelters.csv",
                            usecols=['X', 'Y'], index_col=None)
        data.columns = ['Long', 'Lat']
        data = data.dropna(axis=0)
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="HOMELESS")
        return cleaned_data


    # Returns a dataframe containing the X,Y coordinates
    def CleanSchoolsData():
        data = pd.read_csv(globvars.RAW_DATA_FILEPATH + "schools.csv",
                            usecols=['LATITUDE', 'LONGITUDE'], index_col=None)
        data.columns = ['Lat', 'Long']
        data = data.dropna(axis=0)
        cleaned_data = data.apply(DataCleaner.ConvertToXY, axis=1, feature="SCHOOL")
        return cleaned_data
