"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

from CleanRawData import DataCleaner
import res.globvars as globvars

import sys
import pandas as pd


def main():
    CRIME_FILE, CITY_FILE = DataCleaner.CleanRawData()
    try:
        crime_data = pd.read_csv(CRIME_FILE)
    except Exception as e:
        print(e)
        sys.exit(1)
    
    # Do the analysis...
    
if __name__ == "__main__":
    main()
