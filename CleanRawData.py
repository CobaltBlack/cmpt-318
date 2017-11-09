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
#            city_data = city_data.append(DataCleaner.CleanParksData(), ignore_index=True)
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
        utm_coors = utm.from_latlon(x['Lat'], x['Long'])
        # Vancouver is located in zone 10, the value should not change
        assert utm_coors[2] == 10
        return pd.Series({'X': utm_coors[0], 'Y': utm_coors[1], 'TYPE': feature})

    
    # Returns a dataframe containing the X,Y coordinates for each rapid transit station
    def CleanRTSData():
        rts = pd.read_csv(globvars.RAW_DATA_FILEPATH + "rapid_transit_stations.csv",
                          usecols=['X','Y'])
        rts.columns = ['Long', 'Lat']
        cleaned_rts = rts.apply(DataCleaner.ConvertToXY, axis=1, feature="RTS")
        return cleaned_rts     
    
    
    # Returns a dataframe containing the X,Y coordinates of each park
    def CleanParksData():
        parks = pd.read_csv(globvars.RAW_DATA_FILEPATH + "parks.csv",
                            usecols=["GoogleMapDest"], index_col=None)
        # TODO Change GoogleMapDest to Lat Long column
        cleaned_parks = parks.apply(DataCleaner.ConvertToXY, axis=1, feature="PARK")
        return cleaned_parks