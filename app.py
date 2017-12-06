"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

from CleanRawData import DataCleaner
import res.globvars as globvars

import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KDTree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.externals import joblib

from scipy.stats import chi2_contingency
from scipy.stats import levene

from statsmodels.stats.multicomp import pairwise_tukeyhsd


crime_color_map = {
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
    'GARDEN': 'g',
    'HOMELESS': 'magenta',
    'CITYPROJECT': 'c',
    'SCHOOL': 'orange'
}


def PrintHeaderRule():
    print('\n\n#############################')

#
# Plots two 2D scatter plots of the location of the crimes and city features
# No legend is given as the data is very dense when all put into a single graph
# Designed as a visual aid and not for analysis purposes
#
def PlotAllData(crime_data, city_data):
    crime_data['color'] = crime_data['TYPE'].apply(lambda x: crime_color_map[x])
    plt.figure(figsize=(10,10))
    plt.scatter(crime_data['X'], crime_data['Y'], c=crime_data['color'],  s=1)
    plt.savefig('crime.jpg')
    
    city_data['color'] = city_data['TYPE'].apply(lambda x: city_color_map[x])
    plt.figure(figsize=(10,10))
    plt.scatter(city_data['X'], city_data['Y'], c=city_data['color'],  s=1)
    plt.savefig('city.jpg')

#
# A visualizetion tool to view the distribution of city features around a crime
# crime_data, city_data: the dataframes containing all of the data used
# crimetype: a single type of crime to be plotted
# features: a list of city feature types to be plotted
#
def PlotCrimeVsFeatures(crime_data, city_data, crimetype, features):
    crimes = crime_data[crime_data['TYPE'] == crimetype]
    
    plt.figure(figsize=(15,15))
    # Since the crimes are much more dense, make them less visible on the graph
    plt.scatter(crimes['X'], crimes['Y'], c='black', 
                s=3, alpha=0.5, label=crimetype)
    #
    # For each feature, add all the locations on to the map, making sure to 
    # add a label and unique color to enhance visibility
    #
    for feature in features:
        to_plot = city_data[city_data['TYPE'] == feature]
        plt.scatter(to_plot['X'], to_plot['Y'], c=city_color_map[feature], 
                s=30, label=feature)
    plt.legend(loc=2)
    # Create a relevant filename so multiple images can be produced
    filename = (crimetype + '_' + '_'.join(features) + '.png')
    filename = re.sub('[/ ]','-', filename)
    plt.savefig(filename)
    
#
# Performs a Tukey Pairwise test on the distances of each type
# of crime to a given city feature
#
def TukeyHelper(crime_data, crimes, feature, alpha = 0.05):
    df = crime_data.loc[crime_data['TYPE'].isin(crimes)]
    # The distance data is all right skewed as most crimes are often somewhat
        # near city features, so we sqrt them to get normal looking data
    df['nearest_'+feature] = np.sqrt(df['nearest_'+feature])
    # Plot the histogram of the data to confirm that it is mostly normal
    groups = df.groupby('TYPE')
    for crime in crimes:
        plt.hist(groups.get_group(crime)['nearest_'+feature], 
                 alpha=0.5,
                 label=crime)
    plt.legend(loc=1)
    plt.show()
    # Run Levene test on each crime distance distribution
    # https://stackoverflow.com/questions/26202930/pandas-how-to-apply-scipy-stats-test-on-a-groupby-object
    values_per_group = [np.sqrt(col) for col_name, col in groups['nearest_'+feature]]
    print('P-value for Levene test:', levene(*values_per_group).pvalue)
    # Finally run the Tukey test, with the above data confirming / denying
        # the test's validity
    posthoc = pairwise_tukeyhsd(df['nearest_'+feature], df['TYPE'], alpha=alpha)
    print(posthoc)

#
# Determine if the mean distance to each city feature
# from each type of crime varies
#
def DistanceToCityFeatureTukey(crime_data, city_data):
    PrintHeaderRule()
    print('Tukey tests on mean distance to each city feature\n')
    #
    # We are doing a test for each city feature type, adjust for this in 
    # the statistical analysis
    #
    alpha = 0.05 / 5
    
    #    
    # For each city feature, determine which crimes are closer
    # Note that we have limited the crimes to compare in order to increase 
    # robustness to error and ignore data we don't really care about
    #
    print('RTS TUKEY:')
    TukeyHelper(crime_data, [ 'Break and Enter Residential/Other',
                              'Break and Enter Commercial',
                              'Other Theft',
                              'Theft of Vehicle',
                              'Theft of Bicycle' ], 
                          'RTS', alpha)
    
    print('COMMUNITY CENTER TUKEY:')
    TukeyHelper(crime_data, [ 'Mischief',
                              'Break and Enter Residential/Other',
                              'Other Theft',
                              'Theft of Vehicle',
                              'Theft of Bicycle',
                              'Theft from Vehicle' ], 
                'COMMUNITYCENTER', alpha)
    
    print('HOMELESS SHELTER TUKEY:')
    TukeyHelper(crime_data, globvars.USABLE_CRIMES, 'HOMELESS', alpha)
    
    print('SCHOOL TUKEY:')
    TukeyHelper(crime_data, [ 'Mischief',
                              'Break and Enter Residential/Other',
                              'Other Theft',
                              'Theft of Vehicle',
                              'Theft of Bicycle',
                              'Theft from Vehicle' ], 
                'SCHOOL', alpha)
    
    print('GARDEN TUKEY:')
    TukeyHelper(crime_data, [ 'Mischief',
                              'Break and Enter Residential/Other',
                              'Other Theft',
                              'Theft of Vehicle',
                              'Theft from Vehicle' ], 
                'GARDEN', alpha)


#
# Run a chi-squared test on the number of each type of crime that occurs
# within the "nearby" raidus of each city feature
# Assumes "nearby" crimes have already been calculated
#
def ChiTests(crime_data, city_data):
    PrintHeaderRule()
    print('Chi-squared test on all data\n')
    # Contingency table where rows are crimes and columns are the city features
    # Each entry indicates how many crimes occuur within the radius of the feature
    observed = []
    for crime_type in globvars.USABLE_CRIMES:
        crime_type_counts = city_data.groupby('TYPE')['nearby_count_' + crime_type].sum()
        observed.append(list(crime_type_counts.values))

    observed = np.array(observed)
    _, p, _, expected = chi2_contingency(observed)
    print('p-value of chi-sqaured test: {}'.format(p))

    print('Percentage of deviation from expected independent values:')
    np.set_printoptions(threshold=np.nan)
    print(np.around((observed - expected)/expected*100))
    # Reset default
    np.set_printoptions(threshold=1000)


#
# Run a chi-squared test comparing one crime to all other types of crimes
# against a single city feature
#
def ChiTestOneCrimeOneFeature(crime_data, city_data, crimetype, feature):
    PrintHeaderRule()
    print('Chi-squared test on', crimetype, 'and', feature, '\n')
    #
    # A 2x2 contingency table, where the first row is the crime of interest
    # the scond row is all other crimes, the first column is "occurs near" the
    # city feature of interest, and the second column is "does not occur neear"
    #
    observed = []
    idx = crime_data['TYPE'] == crimetype;
    crimes_of_type = crime_data.loc[idx]
    near_feature = sum(crimes_of_type['nearby_count_' + feature] > 0)
    total = len(crimes_of_type)
    observed.append([near_feature, total-near_feature])
    
    crimes_not_of_type = crime_data.loc[~idx]
    near_feature = sum(crimes_not_of_type['nearby_count_' + feature] > 0)
    total = len(crimes_not_of_type)
    observed.append([near_feature, total-near_feature])
    
    _, p, _, expected = chi2_contingency(observed)
    print(observed)
    print(expected)
    print(p)
    
#
# Convert the count of city features into a binary value of 0 or 1
#  0 means there is no city feature nearby, 1 means there is at least 1
#
def BinarizeCounts(data):
    def f(x): 
        return min(1, x)
    binarize = np.vectorize(f)
    output = binarize(data)
    return output
#
# Print the probabiity of each type of crime occurring near an area with a given
#   set of city featuers nearby.
# nearby_city_features: a 2D array where each row is a test to run consisting
#   of 5 numbers indiciating the number of each city feature
# model: the classification model to evaluate on
# nearby_count_columns: The list of crimes used in evaluation
# crime_percentages: The overall crime rate used in calculating relative scores
#
def PredictNearbyCrimes(nearby_city_features, model, nearby_count_columns, 
                        crime_precentages=None):
    results = model.predict_proba(nearby_city_features)
    classes = model.classes_
    # For each test passed in nearby_city_features, print out the results
    for i in range(len(results)):
        # Print the list of city features we want to find the crimes for
        city_features = nearby_city_features[i]
        print('\n')
        for j in range(len(city_features)):
            print('{} {}'.format(city_features[j], 
                  nearby_count_columns[j].replace('nearby_count_', "")), end='')
            if (j != len(city_features) - 1):
                print(', ', end="")
            else:
                print(':')

        # Print the liklihood of each type of crime occuring
        result_df = pd.DataFrame(data={'Crime': classes, 
                                    'Likelihood': results[i]*100})
        result_df['Likelihood'] = result_df['Likelihood'].apply(round, args=[2])
        # If the overall crime rates were provided, calculate the relative liklihood
        if crime_precentages != None:
            result_df['Relative Likelihood'] = result_df.apply(
                lambda x: (x['Likelihood'] - crime_precentages[x['Crime']] ) /\
                                crime_precentages[x['Crime']] * 100
            , axis=1)
        print(result_df.sort_values('Likelihood', ascending=False))

#
# Generate Naive Bayes and SVM classifiers to predict the frequency of each type
# of crime given a list of nearby city features
#
def ClassifyCrimeTypes(crime_data, city_data):
    # Generate a list of column names from each city feature type
    nearby_count_columns = \
        list(map(lambda x: 'nearby_count_' + x, globvars.CITY_FEATURE_TYPES))
    # Remove those we don't care about
    nearby_count_columns.remove('nearby_count_PARK')
    nearby_count_columns.remove('nearby_count_CITYPROJECT')
    # Split the data into features and class
    X = crime_data[nearby_count_columns]
    y = crime_data['TYPE']
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Train a Naive Bayes classifier
    bayes_model = make_pipeline(FunctionTransformer(BinarizeCounts), 
                                GaussianNB())
    bayes_model.fit(X_train, y_train)
    
    # Report accuracy
    PrintHeaderRule()
    print('Classifier models accuracy:')
    print('\nBayes Model Score: {}'.format(bayes_model.score(X_test, y_test)))
    
    # Train an SVM classifier + cache it to avoid long training times each run
    if os.path.isfile(globvars.SVM_PICKLE):
        svm_model = joblib.load(globvars.SVM_PICKLE)
    else:
        svm_model = make_pipeline(SVC(probability=True))
        print('######################\n' + \
              '# BEGIN TRAINING SVM #\n' + \
              '######################\n')
        svm_model.fit(X_train, y_train)
        joblib.dump(svm_model, globvars.SVM_PICKLE)
        print('Done!')
    # Report accuracy
    print('\nSVM Model Score: {}'.format(svm_model.score(X_test, y_test)))
    
    # Get the overall criem rate
    crime_precentages = {}
    crime_counts = crime_data.groupby('TYPE')
    for name, group in crime_counts:
        crime_precentages[name] = \
            round(group.X.count() / crime_data.X.count() * 100, 2)
    
    # Predict likelihood of crimes based on nearby city features
    city_features_query = [
        [1,0,0,0,0],
        [0,1,0,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,0,0,0,1]
    ]
    
    PrintHeaderRule()
    print('Naive Bayes predictions:')
    PredictNearbyCrimes(city_features_query, bayes_model, nearby_count_columns,
                        crime_precentages=crime_precentages)
    
    PrintHeaderRule()
    print('SVM predictions:')
    PredictNearbyCrimes(city_features_query, svm_model, nearby_count_columns, 
                        crime_precentages=crime_precentages)
    
    return bayes_model, svm_model


# Perform various calculations on the data to be used in later testing
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
    PrintHeaderRule()
    print('Number of crimes\n')
    for name, group in crime_counts:
         print('\t{}: {} ({}%)'.format(name, group.X.count(), \
                           round(group.X.count() / crime_data.X.count() * 100,2)))
    
    # Uncomment if you would like to see Tukey tests
#    DistanceToCityFeatureTukey(crime_data, city_data)
    
    # For each crime type, find the average distance from each city feature type
    PrintHeaderRule()
    print('Average distance to each city feature per crime')
    for crime_type in globvars.USABLE_CRIMES:
        mean_dist = city_data.groupby('TYPE')['avg_dist_' + crime_type].mean()
        print("\n\nAverage distances for crime:", crime_type)
        print(mean_dist)

    # Chi1 Contingency test for types of crimes happening nearby a city feature
    ChiTests(crime_data, city_data)
   
    # Try classifying crime types based on nearby city features
    ClassifyCrimeTypes(crime_data, city_data)


    PlotCrimeVsFeatures(crime_data, city_data, 'Break and Enter Residential/Other',
                        ['GARDEN', 'HOMELESS', 'RTS', 'SCHOOL'])
    PlotCrimeVsFeatures(crime_data, city_data, 'Theft of Vehicle',
                        ['RTS', 'SCHOOL'])
    PlotCrimeVsFeatures(crime_data, city_data, 'Other Theft',
                        ['RTS', 'COMMUNITYCENTER'])

if __name__ == "__main__":
    main()
