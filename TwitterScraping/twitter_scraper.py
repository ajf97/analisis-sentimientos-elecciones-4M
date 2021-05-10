__author__ = "Alejandro Jerónimo"

# Import libraries

import tweepy as tw
import pandas as pd
import json

consumer_key = "tn5h2e8ZQCZstJWChmvgD1EVO"
consumer_secret = "6tevkUQXL1t2LvHjKv7mofK3SCo1wHJhSKI2VYeXEqB97d0T2z"
access_token = "400222622-Hgk5MrTBrXNuz4SAFy6kNDychrxXQv0wljx2vy2D"
access_token_secret = "16uwgKrGHzwJv8jsyfn7GNAM0xDzW5qOcvXSfM3Ci7Ttq"

# API authentication

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tw.API(auth, wait_on_rate_limit=True,
 wait_on_rate_limit_notify=True)


# Define the topic search and date variables
search_topic = "#Elecciones4M"
date_since = "2021-05-04"

# Collect tweets
tweets = tw.Cursor(api.search, q=search_topic, lang="es",
since=date_since).items(5)

tweets_list = [tweet.text for tweet in tweets]

for t in tweets_list:
    print(t)