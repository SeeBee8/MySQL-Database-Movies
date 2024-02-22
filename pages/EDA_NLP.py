#st.image('images/Logobue_sqTBDM.svg')
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import json
from wordcloud import WordCloud
from nltk import casual_tokenize
# Changing the Layout
st.set_page_config( layout="wide")

with open('config/filepaths.json') as f:
    FPATHS = json.load(f)
@st.cache_data
def load_data(fpath):
    return joblib.load(fpath)


df = load_data(FPATHS['data_NLP']['nlp_df'])
df.head(2)

    
st.title("Exploratory Data Analysis of Movie Reviews")

st.subheader("Word Cloud")
# select which version of wordclouds
wc_choice = st.radio("Select WordCloud Text: ", ["Raw Text",'Lemmas'], index=0, horizontal=True)
wc_choice


if wc_choice=='Lemmas':
    fpath_wc = FPATHS['eda']['wordclouds-lemmas']
else:
    fpath_wc = FPATHS['eda']['wordclouds-raw']
fpath_wc
st.image(fpath_wc)

st.divider()

st.subheader("Frequency Distribution")
fpath_freq = FPATHS['eda']['freq']
st.image(fpath_freq)



