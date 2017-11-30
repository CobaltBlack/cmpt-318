"""

Copyright (c) 2017 Joshua McManus, Eric Liu

"""

RESOURCE_FILES_PATH = "./res/files/"
RAW_DATA_FILEPATH = "./res/raw_data/"
SVM_PICKLE = "./res/crime_svm.pkl"
USABLE_CRIMES = [
    'Mischief',
    'Theft from Vehicle' ,
    'Break and Enter Residential/Other',
    'Theft of Vehicle',
    'Break and Enter Commercial',
    'Other Theft',
    'Theft of Bicycle'
]

CITY_FEATURE_TYPES = [
    'RTS',
    'PARK',
    'COMMUNITYCENTER',
    'GARDEN',
    'HOMELESS',
    'CITYPROJECT',
    'SCHOOL'
]

STANLEY_PARK_Y = 5460250
LOCALITY_RADIUS = 500