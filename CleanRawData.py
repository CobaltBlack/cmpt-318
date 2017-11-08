"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

import res.globvars as globvars
import pandas as pd
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
        
        crimes.to_csv(globvars.RESOURCE_FILES_PATH + "crime_data.csv", index=False)
