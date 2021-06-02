__author__ = "Alejandro Jerónimo"

# %% Import libraries

import numpy as np
import pandas as pd
import tweepy as tw

import credentials as cr

# %% API setup


def setup():
    """
    Setup the Twitter's API with the keys provided in credentials.py
    """

    # Authentication
    auth = tw.OAuthHandler(cr.CONSUMER_KEY, cr.CONSUMER_SECRET)
    auth.set_access_token(cr.ACCESS_TOKEN, cr.ACCESS_TOKEN_SECRET)

    api = tw.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    return api


def search_topic(api, topic, items, date_from=None, date_to=None):
    # Collect tweets
    tweets = tw.Cursor(
        api.search_30_day,
        environment_name="dev",
        query=topic,
        fromDate=date_from,
        toDate=date_to,
    ).items(items)

    return tweets


# %% Define the topic search and date variables
topic = "#Elecciones4M"
date_from = "202105040900"
date_to = "202105042300"
n_items = 1000

api = setup()
tweets = search_topic(api, topic, n_items, date_from, date_to)
tweets = [tweet for tweet in tweets]
# %% Create dataframe

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=["Tweets"])


data["len"] = np.array([len(tweet.text) for tweet in tweets])
data["ID"] = np.array([tweet.id for tweet in tweets])
data["Date"] = np.array([tweet.created_at for tweet in tweets])
data["Source"] = np.array([tweet.source for tweet in tweets])
data["Likes"] = np.array([tweet.favorite_count for tweet in tweets])
data["RTs"] = np.array([tweet.retweet_count for tweet in tweets])

# %% Export dataframe to csv

data.to_csv("../../data/raw/data.csv", index=False, header=True)
