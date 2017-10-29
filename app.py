# -*- coding: utf-8 -*-
"""

Copyright (c) 2017 Joshua McManus, Eric Liu
This applications and its creators have no affiliation with Valve or Steam.

"""
import json
import requests

API_FILE = './api-key.txt'
API_KEY_LENGTH = 32

STEAM_API_DOMAIN = 'http://api.steampowered.com/'

# Makes a call to the Steam API to retrieve all Steam apps
#  and saves it to a json file
def FetchAppList():
    req_url = (
            STEAM_API_DOMAIN + 
            'ISteamApps/GetAppList/v0002'
            )
    steamapps = requests.get(req_url)
    with open('./apps.json', 'w') as f:
        json.dump(steamapps.json(), f)
        f.close

def FetchSupportedApiCalls(API_KEY):
    req_url = (
           STEAM_API_DOMAIN + 
           'ISteamWebAPIUtil/GetSupportedAPIList/v0001'
           )
    apis = requests.get(req_url, params={'key': API_KEY})
    with open('./apis.json', 'w') as f:
        json.dump(apis.json(), f)
        f.close

def main():
    # Fetch the API key for Steam from the root directory
    apiFile = open(API_FILE, 'r')
    API_KEY = apiFile.read().strip()
    assert len(API_KEY) == API_KEY_LENGTH

    #FetchAppList();
    #FetchSupportedApiCalls(API_KEY)

if __name__ == "__main__":
    main()
