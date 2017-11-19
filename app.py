"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

from CleanRawData import DataCleaner
import res.globvars as globvars

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.stats import *

color_map = {
    'Mischief': 'b',
    'Theft from Vehicle': 'r',
    'Break and Enter Residential/Other': 'y',
    'Theft of Vehicle': 'g',
    'Break and Enter Commercial': 'tab:purple',
    'Other Theft': 'xkcd:sky blue',
    'Theft of Bicycle': 'c'
}

city_color_map = {
    'RTS': 'b',
    'PARK': 'r',
    'COMMUNITYCENTER': 'y',
    'GARDEN': 'tab:purple',
    'HOMELESS': 'xkcd:sky blue',
    'CITYPROJECT': 'c',
    'SCHOOL': 'k'
}

def PlotData(crime_data, city_data):
    crime_data['color'] = crime_data['TYPE'].apply(lambda x: color_map[x])
    plt.figure(figsize=(25,25))
    plt.scatter(crime_data['X'], crime_data['Y'], c='blue',  s=1)
    plt.savefig('crime.jpg')
    
    city_data['color'] = city_data['TYPE'].apply(lambda x: city_color_map[x])
    plt.figure(figsize=(25,25))
    plt.scatter(city_data['X'], city_data['Y'], c='blue',  s=1)
    plt.savefig('city.jpg')

def main():
    CRIME_FILE, CITY_FILE = DataCleaner.CleanRawData()
    try:
        crime_data = pd.read_csv(CRIME_FILE)
        city_data = pd.read_csv(CITY_FILE)
    except Exception as e:
        print (e)
        sys.exit(1)

    crime_kd = KDTree(crime_data[['X', 'Y']])
    dist, ind = crime_kd.query(city_data[['X','Y']].values, k=50)
    # Ignore crimes that are too far away from the feature??
    city_data['crimes_ind'] = pd.Series(list(ind))
    city_data['crimes_dist'] = pd.Series(list(dist))
    
    # Find the avergae distance to crimes per city feature type
    city_data['avg_dist'] = city_data['crimes_dist'].apply(lambda x: sum(x)/len(x))
    mean_dist = city_data.groupby('TYPE').avg_dist.mean()
        
    # Get average distance to crime of each type
    for crime_type in globvars.USABLE_CRIMES:
        curr_crime_data = crime_data[crime_data['TYPE'] == crime_type]
        curr_crime_kd = KDTree(curr_crime_data[['X', 'Y']])
        dist, ind = curr_crime_kd.query(city_data[['X','Y']].values, k=3)
        
        # Ignore crimes that are too far away from the feature
        city_data[crime_type + '_ind'] = pd.Series(list(ind))
        city_data[crime_type + '_dist'] = pd.Series(list(dist))
        
        # Find the avergae distance to crimes per city feature type
        city_data[crime_type + '_avg_dist'] = city_data[crime_type + '_dist'].apply(lambda x: sum(x)/len(x))
        
        mean_dist = city_data.groupby('TYPE')[crime_type + '_avg_dist'].mean()
#        print("\n\navg dist for crime", crime_type)
#        print(mean_dist)
            
    # For each crime, get closest city features
    city_kd = KDTree(city_data[['X', 'Y']])
    dist, ind = city_kd.query(crime_data[['X','Y']].values, k=5)
    crime_data['city_ind'] = pd.Series(list(ind))
    crime_data['city_dist'] = pd.Series(list(dist))
    
    # Find the average distance to crimes per city feature type
    crime_data['avg_dist'] = crime_data['city_dist'].apply(lambda x: sum(x)/len(x))
    mean_dist_crime = crime_data.groupby('TYPE').avg_dist.mean()

    print(mean_dist_crime)
    print(city_data.columns)
    print(crime_data.columns)
        
        
if __name__ == "__main__":
    main()
