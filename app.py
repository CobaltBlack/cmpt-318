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

def main():
    # Fetch the API key for Steam from the root directory
    apiFile = open(API_FILE, 'r')
    API_KEY = apiFile.read().strip()
    assert len(API_KEY) == API_KEY_LENGTH

    supported_apis = (
            STEAM_API_DOMAIN + 
            'ISteamWebAPIUtil/GetSupportedAPIList/v0001'
            )
    resp = requests.get(supported_apis, params={'key': API_KEY})
    
    API_CALLS = resp.json()

if __name__ == "__main__":
    main()
