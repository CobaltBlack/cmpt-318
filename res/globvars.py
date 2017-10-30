"""

Copyright (c) 2017 Joshua McManus, Eric Liu
This applications and its creators have no affiliation with Valve or Steam.

"""
import json

APPLIST = []
API_COMMANDS = []

def init():
   global APPLIST
   appFile = open('./res/files/apps.json', 'r')
   APPLIST = json.load(appFile)
   appFile.close

   global API_COMMANDS
   apiFile = open('./res/files/apis.json', 'r')
   API_COMMANDS = json.load(apiFile)
   apiFile.close
