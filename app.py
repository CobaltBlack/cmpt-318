"""

Copyright (c) 2017 Joshua McManus, Eric Liu
This applications and its creators have no affiliation with Valve or Steam.

"""

#from res.globvars import APPLIST
from CleanCrimeData import CleanCrimeData
from res.globvars import RESOURCE_FILES_PATH

import os
import sys
import pandas as pd

CRIME_FILE = RESOURCE_FILES_PATH + "crime_data.csv"

def main():
    if not os.path.isfile(CRIME_FILE):
        CleanCrimeData(RESOURCE_FILES_PATH + 'crime_csv_all_years.csv', CRIME_FILE)
    try:
        crime_data = pd.read_csv(CRIME_FILE)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    # Do the analysis...
    
if __name__ == "__main__":
    main()
