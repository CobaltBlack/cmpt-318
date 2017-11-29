"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

from CleanRawData import DataCleaner
import res.globvars as globvars

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
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


# Performs a Tukey Pairwise test on the distances of each type of crime to a
#  given city feature
def FeatureDistanceHelper(crime_data, crimes, feature, alpha = 0.05):
    df = crime_data.loc[crime_data['TYPE'].isin(crimes)]
    posthoc = pairwise_tukeyhsd(df['nearest_'+feature], df['TYPE'], alpha=alpha)
    print(posthoc)

# Find the average distance to each type of city feature from each type of crime
#   Given all crimes, find the distance to the nearest of each type of feature
def CrimeDistanceFeatureType(crime_data, city_data):
    # We are doing a test for each city feature type, adjust for this in 
    #  the statistical analysis
    alpha = 0.05 / 7
    
    # For each city feature, determine which crimes are closer
    # Note that we have limited the crimes to compare in order to increase 
    #  robustness to error and ignore data we don't really care about
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


# Analyze crimes within a radius of each city feature
def ChiTests(crime_data, city_data):
    #
    # Sum nearby crime types for each city feature type
    # e.g. Total number of 'thefts of bicyles' near 'school'
    observed = []
    for crime_type in globvars.USABLE_CRIMES:
        crime_type_counts = city_data.groupby('TYPE')['nearby_count_' + crime_type].sum()
        observed.append(list(crime_type_counts.values))
#        print(crime_type_counts)

    np.set_printoptions(threshold=np.nan)

    # Contingency table test
    observed = np.array(observed)
    _, p, dof, expected = chi2_contingency(observed)
    print('p-value of chi-sqaured test: {}'.format(p))

    print('Percentage of deviation from expected independent values:')
    print(np.around((observed - expected)/expected*100))


# Convert the count of city features into a binary value of 0 or 1
#  0 means there is no city feature nearby, 1 means there is at least 1
# This will allow for easier testing
def BinarizeCounts(data):
    def f(x): 
        return min(1, x)
    binarize = np.vectorize(f)
    return binarize(data)


def ClassifyCrimeTypes(crime_data, city_data):
    
    # Split crime data into feature and class
    nearby_count_columns = list(map(lambda x: 'nearby_count_' + x, globvars.CITY_FEATURE_TYPES))
    X = crime_data[nearby_count_columns]
    y = crime_data['TYPE']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    #
    # Train a Naive Bayes classifier to 
    # guess crime type based on nearby features
    #
    bayes_model = make_pipeline(FunctionTransformer(BinarizeCounts), 
                                    GaussianNB())
    bayes_model.fit(X_train, y_train)
    
    # Report accuracy
    print(bayes_model.score(X_test, y_test))
    print(bayes_model.predict_proba([0,1,0,0,0,0,0]))
    
    #
    # Train a SVM to classify crime type based on number of nearby features
    #
    svm_model = SVC(probability=True)
    print('######################\n' + \
          '# BEGIN TRAINING SVM #\n' + \
          '######################')
    svm_model.fit(X_train, y_train)
    print('SVM Model Score: {}'.format(svm_model.score(X_test, y_test)))
    print(svm_model.predict_proba([0,1,0,0,0,0,0]).reshape(1, -1))


def CalculateDistances(crime_data, city_data):
    # Get the k nearest crimes per city feature
    crime_kd = KDTree(crime_data[['X', 'Y']])
    dist, ind = crime_kd.query(city_data[['X','Y']].values, k=50)
    city_data['nearest_crimes_ind'] = pd.Series(list(ind))
    city_data['nearest_crimes_dist'] = pd.Series(list(dist))
    
    # Get the k nearest city fetures per crime
    city_kd = KDTree(city_data[['X', 'Y']])
    dist, ind = city_kd.query(crime_data[['X','Y']].values, k=5)
    crime_data['nearest_city_ind'] = pd.Series(list(ind))
    crime_data['nearest_city_dist'] = pd.Series(list(dist))
    # Find the average distance to crimes per city feature type
    
    city_features = city_data.groupby('TYPE')
    for name, group in city_features:
        # Collect the distances to each of the nearest city feature per crime
        feature_kd = KDTree(group[['X', 'Y']])
        dist, ind = feature_kd.query(crime_data[['X', 'Y']].values, k=1)
        crime_data['nearest_' + name] = pd.Series(item[0] for item in dist)  
        # Collect the number of nearby city features to each crime
        #  Use a predefined radius to determine what is "nearby"
        counts = feature_kd.query_radius(crime_data[['X','Y']].values, 
                                         globvars.LOCALITY_RADIUS, 
                                         count_only=True)
        crime_data['nearby_count_' + name] = pd.Series(list(counts))
    
    for crime_type in globvars.USABLE_CRIMES:
        # Get average distance to crime of each type
        curr_crime_data = crime_data[crime_data['TYPE'] == crime_type]
        curr_crime_kd = KDTree(curr_crime_data[['X', 'Y']])
        dist, ind = curr_crime_kd.query(city_data[['X','Y']].values, k=3)

        city_data['nearest_ind_' + crime_type] = pd.Series(list(ind))
        city_data['nearest_dist_' + crime_type] = pd.Series(list(dist))

        # Find the average distance to crimes per city feature type
        city_data['avg_dist_' + crime_type] = \
             city_data['nearest_dist_' + crime_type].apply(lambda x: sum(x)/len(x))

        # Get number of crimes within a radius
        counts = curr_crime_kd.query_radius(city_data[['X','Y']].values, globvars.LOCALITY_RADIUS, count_only=True)
        city_data['nearby_count_' + crime_type] = pd.Series(list(counts))


def main():
    CRIME_FILE, CITY_FILE = DataCleaner.CleanRawData()
    try:
        crime_data = pd.read_csv(CRIME_FILE)
        city_data = pd.read_csv(CITY_FILE)
    except Exception as e:
        print (e)
        sys.exit(1)

    CalculateDistances(crime_data, city_data)
    
    crime_counts = crime_data.groupby('TYPE')
    print('Number of crimes:')
    for name, group in crime_counts:
         print('\t{}: {} ({}%)'.format(name, group.X.count(), \
                           round(group.X.count() / crime_data.X.count() * 100,2)))
    
#    CrimeDistanceFeatureType(crime_data, city_data)
    
    # For each crime type, find the average distance from each city feature type
#    for crime_type in globvars.USABLE_CRIMES:
#        mean_dist = city_data.groupby('TYPE')['avg_dist_' + crime_type].mean()
#        print("\n\nAverage distances for crime:", crime_type)
#        print(mean_dist)

    # Chi1 Contingency test for types of crimes happening nearby a city feature
#    ChiTests(crime_data, city_data)
   
    # Try classifying crime types based on nearby city features
#    ClassifyCrimeTypes(crime_data, city_data)


if __name__ == "__main__":
    main()
