# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <center><img src="images/banner.jpg"></center>
#
# <center><h1>Análisis de sentimientos de las elecciones a la Asamblea de Madrid 2021</h1></center>
#
#

# %% [markdown]
# ## Introducción
#
# El pasado martes 4 de mayo se celebraban las elecciones a la Asamblea de Madrid 2021. Estas elecciones son de alguna forma excepcionales y polémicas desde el mismo instante de su convocatoria. Esto se debe a la pandemia y a la polarización política de los últimos años en España. Así pues, se han saldado con la victoria de los partidos derecha y la derrota de la izquierda, obligando a dos de los principales candidatos a abandonar la política. También hemos asistido a la desaparición del partido Ciudadanos de la Asamblea, empeorando aún más su crisis como partido.
#
# Por si fuera poco, la campaña electoral se desarrolló bajo un clima de mucha tensión, ya que algunos políticos del país recibieron sobres con balas. En las redes sociales, esta polarización se manifiesta más. Es frecuente ver insultos y descalificaciones en la mayoría de los mensajes.
#
# Las elecciones de Madrid pueden afectar de alguna forma a elecciones futuras. Por tanto, es interesante analizar los sentimientos de los usuarios en redes —en este caso Twitter— con el objetivo de ver cómo se desarrolla esta polarización en todo el país.
#
# ### 🟢 Objetivo:
#
# > Realizar un análisis de sentimientos en Twitter para ver el desarrollo de la polarización en España.
#
# ### 📚 Librerías
#
# Para conseguir nuestro objetivo, necesitamos las siguientes librerías:
#
# * **Tweepy**: la usaremos para descargar los tweets utilizando la API de Twitter.
# * **Pandas**: librería para el análisis de los datos.

# %%
import random
from string import punctuation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from emoji.unicode_codes.es import EMOJI_UNICODE_SPANISH
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from PIL import Image
from sentiment_analysis_spanish import sentiment_analysis
from wordcloud import WordCloud

# Tienes que descargarte las stopwords primero usando nltk.download()

# %% [markdown]
# ## TODO: Extracción de tweets

# %% [markdown]
# ## 🧹 Limpieza de datos
#
# Una vez extraídos los tweets en un archivo CSV, el siguiente paso consiste en eliminar aquellos datos que no son importantes para el análisis. Para ello, vamos a usar la librería `Pandas`, que nos permitirá gestionar los tweets en tablas llamadas *Dataframes*.

# %%
# Importamos los datos a un Dataframe
data = pd.read_csv("data/data.csv", index_col="ID")

data.head()

# %% [markdown]
# A continuación, vamos a hacer un análisis exploratorio para ver los datos que disponemos. Comprobaremos también si tenemos datos nulos y duplicados.

# %%
# Mostramos las filas y columnas del dataframe
data.shape

# %%
# Comprobamos valores nulos
data.isna().any()

# %%
# Eliminar la cadena 'RT' en los tweets para quedarnos con la información relevante
data["Tweets"] = data["Tweets"].str.replace(
    r"RT.*:",
    "",
    regex=True,
)

# %%
# Eliminar saltos de línea y retorno para tenerlo todo en una frase
data["Tweets"] = data["Tweets"].str.replace("\n", "")
data["Tweets"] = data["Tweets"].str.replace("\r", "")


# %%
# Eliminar etiqueta #Elecciones4M
data["Tweets"] = data["Tweets"].str.replace("#Elecciones4M", "")

# %%
# Eliminar links
links_index = data[data["Tweets"].str.contains(r"//t\.co.*", regex=True)].index
data = data.drop(links_index)

# %% [markdown]
# Seguimos haciendo limpieza de los datos, eliminando información que no es relevante para el análisis.

# %%
# Seguimos eliminando links
links_index = data[data["Tweets"].str.contains("//t.c…")].index
data = data.drop(links_index)
data.shape

# %%
# Eliminamos filas con ruido
data = data.drop(
    [
        1389714946599137283,
        1389714670081265669,
        1389714532499697667,
        1389715033786040322,
        1389714949367336965,
        1389714934095880192,
        1389714915737448451,
        1389714894241554438,
        1389714495413555202,
        1389715170969231361,
        1389715027356266502,
    ]
)

# %%
# Eliminar duplicados
data = data.drop_duplicates(subset=["Tweets"])

# %%
# Exportamos los datos preparados
data.to_csv("data/data_prepared.csv")

# %% [markdown]
# ## Procesamiento del texto
#
# Después de limpiar los datos, tenemos que hacer el procesamiento del texto.
#

# %%
spanish_stopwords = stopwords.words("spanish")

# Transformamos cada palabra en su raíz
stemmer = SnowballStemmer("spanish")

# Eliminamos signos puntuación, números y emojis
emoji_list = list(EMOJI_UNICODE_SPANISH.values())

non_words = list(punctuation)
non_words.extend(["¿", "?"])
non_words.extend(map(str, range(10)))
non_words.extend(emoji_list)

# %%
tweets_text = data["Tweets"]
tweets_text = tweets_text.apply(word_tokenize)

# tweets_text.head()

# %%
# Importamos la imagen del logo de Twitter
twitter_logo = np.array(Image.open("images/twitter_logo.png"))

figure = plt.figure()
figure.set_figwidth(14)
figure.set_figheight(18)

plt.imshow(twitter_logo)
plt.axis("off")
plt.show()


# %%
# Función para generar el color azul de la nube de tweets
def blue_color(
    word,
    font_size,
    position,
    orientation,
    ramdom_state=None,
    **kwargs,
):
    return "hsl(210, 100%%, %d%%)" % random.randint(50, 70)


# %%
tweets_long_string = data["Tweets"].tolist()
tweets_long_string = " ".join(tweets_long_string)


# %%
twitter_wc = WordCloud(
    background_color="white",
    max_words=1500,
    mask=twitter_logo,
)

# Generamos la nube de palabras
twitter_wc.generate(tweets_long_string)

# Mostramos por pantalla la nube de palabras
figure = plt.figure()
figure.set_figwidth(14)
figure.set_figheight(18)

plt.imshow(
    twitter_wc.recolor(color_func=blue_color, random_state=3),
    interpolation="bilinear",
)
plt.axis("off")
plt.show()


# %%
