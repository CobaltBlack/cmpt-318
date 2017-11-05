# -*- coding: utf-8 -*-
"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

import pandas as pd

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

def CleanCrimeData(all_crimes, out_crimes):
    crimes = pd.read_csv(all_crimes, sep=',', header=0,
                         usecols=['TYPE','YEAR','HUNDRED_BLOCK','X','Y'])
    
    crimes = crimes[crimes.YEAR >= MIN_YEAR]
    # The X and Y values are set to zero for any crime against a person to 
    #   protect their identity, which means we can't use the data
    crimes = crimes[(crimes.X != 0.0) | (crimes.Y != 0.0)]
    crimes = crimes[crimes['TYPE'].isin(USABLE_CRIMES)]
    
    crimes.to_csv(out_crimes, index=False)
