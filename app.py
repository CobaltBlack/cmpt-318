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
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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


def FeatureDistanceHelper(crime_data, crimes, feature, alpha = 0.05):
    df = crime_data.loc[crime_data['TYPE'].isin(crimes)]
    posthoc = pairwise_tukeyhsd(df['nearest_'+feature], df['TYPE'], alpha=alpha)
    print(posthoc)

# Find the average distance to each type of city feature from each type of crime
#   Given all crimes, find the distance to the nearest of each type of feature
def CrimeDistanceFeatureType(crime_data, city_data):
    city_features = city_data.groupby('TYPE')
    # Collect the distances to the nearest city feature of each type per crime
    for name, group in city_features:
        feature_kd = KDTree(group[['X', 'Y']])
        dist, ind = feature_kd.query(crime_data[['X', 'Y']].values, k=1)
        crime_data['nearest_' + name] = pd.Series(item[0] for item in dist)
    
    # Aggregate the distances for each type of crime
    #crime_types = crime_data.groupby('TYPE')
    
    alpha = 0.05 / 7
    
    print('CITY PROJECT TUKEY:')
    FeatureDistanceHelper(crime_data, ['Mischief',
                                      'Break and Enter Residential/Other',
                                       'Break and Enter Commercial',
                                      'Other Theft'], 
                          'CITYPROJECT', alpha)
    
    print('RTS TUKEY:')
    FeatureDistanceHelper(crime_data, ['Mischief',
                                      'Break and Enter Residential/Other',
                                       'Break and Enter Commercial',
                                      'Other Theft',
                                      'Theft of Vehicle',
                                      'Theft of Bicycle'], 
                          'RTS', alpha)
    
    print('PARK TUKEY:')
    FeatureDistanceHelper(crime_data, globvars.USABLE_CRIMES, 'PARK', alpha)
    
    print('COMMUNITY CENTER TUKEY:')
    FeatureDistanceHelper(crime_data, ['Mischief',
                                      'Break and Enter Residential/Other',
                                      'Other Theft',
                                      'Theft of Vehicle',
                                      'Theft of Bicycle',
                                      'Theft from Vehicle'], 
                          'COMMUNITYCENTER', alpha)
    
    print('HOMELESS SHELTER TUKEY:')
    FeatureDistanceHelper(crime_data, globvars.USABLE_CRIMES, 'HOMELESS', alpha)
    
    print('SCHOOL TUKEY:')
    FeatureDistanceHelper(crime_data, ['Mischief',
                                      'Break and Enter Residential/Other',
                                      'Other Theft',
                                      'Theft of Vehicle',
                                      'Theft of Bicycle',
                                      'Theft from Vehicle'], 
                          'SCHOOL', alpha)
    
    print('GARDEN TUKEY:')
    FeatureDistanceHelper(crime_data, ['Mischief',
                                      'Break and Enter Residential/Other',
                                      'Other Theft',
                                      'Theft of Vehicle',
                                      'Theft from Vehicle'], 
                          'GARDEN', alpha)
    
def ChiTests(crime_data, city_data):
     # Analyze crimes within a radius of each city feature
    #

    # Sum nearby crime types for each city feature type
    # e.g. Total number of 'thefts of bicyles' near 'school'
    observed = []
    for crime_type in globvars.USABLE_CRIMES:
        crime_type_counts = city_data.groupby('TYPE')[crime_type + '_nearby_count'].sum()
        observed.append(list(crime_type_counts.values))
#        print(crime_type_counts)

    np.set_printoptions(threshold=np.nan)

    # Contingency table test
    observed = np.array(observed)
    _, p, dof, expected = chi2_contingency(observed)
    print('p-value of chi-sqaured test: {}'.format(p))

    print('Percentage of deviation from expected independent values:')
    print(np.around((observed - expected)/expected*100))
    

def main():
    CRIME_FILE, CITY_FILE = DataCleaner.CleanRawData()
    try:
        crime_data = pd.read_csv(CRIME_FILE)
        city_data = pd.read_csv(CITY_FILE)
    except Exception as e:
        print (e)
        sys.exit(1)
    
    CrimeDistanceFeatureType(crime_data, city_data)
    
    return

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

        # Find the average distance to crimes per city feature type
        city_data[crime_type + '_avg_dist'] = city_data[crime_type + '_dist'].apply(lambda x: sum(x)/len(x))
        
        mean_dist = city_data.groupby('TYPE')[crime_type + '_avg_dist'].mean()
#        print("\n\navg dist for crime", crime_type)
#        print(mean_dist)

        # Get number of crimes within a radius
        counts = curr_crime_kd.query_radius(city_data[['X','Y']].values, globvars.LOCALITY_RADIUS, count_only=True)
        city_data[crime_type + '_nearby_count'] = pd.Series(list(counts))


    # For each crime, get closest city features
    city_kd = KDTree(city_data[['X', 'Y']])
    dist, ind = city_kd.query(crime_data[['X','Y']].values, k=5)
    crime_data['city_ind'] = pd.Series(list(ind))
    crime_data['city_dist'] = pd.Series(list(dist))
    
    # Find the average distance to crimes per city feature type
    crime_data['avg_dist'] = crime_data['city_dist'].apply(lambda x: sum(x)/len(x))
    mean_dist_crime = crime_data.groupby('TYPE').avg_dist.mean()

#    print(mean_dist_crime)
#    print(city_data.columns)
#    print(crime_data.columns)

    ChiTests(crime_data, city_data)
   


if __name__ == "__main__":
    main()
