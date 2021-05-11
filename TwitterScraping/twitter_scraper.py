__author__ = "Alejandro Jerónimo"

# Import libraries

from credentials import *
import tweepy as tw
import pandas as pd
import json


# API setup

def setup():

    """
    Setup the Twitter's API with the keys provided in credentials.py
    """

    # Authentication
    auth = tw.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    api = tw.API(auth, wait_on_rate_limit=True,
    wait_on_rate_limit_notify=True)

    return api


def search_topic(api, topic, items, date_from=None, date_to=None):
    # Collect tweets
    tweets = tw.Cursor(api.search_30_day,
    environment_name='dev',
    query=topic,
    fromDate=date_from,
    toDate=date_to).items(items)

    return tweets


# Define the topic search and date variables
topic = "#Elecciones4M"
date_from = "202105040900"
date_to = "202105042300"
n_items = 5

api = setup()
tweets = search_topic(api, topic, n_items, date_from, date_to)
