## Linear regression method for NBA players

#%% Import packages
import numpy as np
import pandas as pd
import sklearn
import plotly.express as px
import streamlit as st
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import linear_model # Linear regression
from sklearn import svm
from sklearn import metrics

#%% Data pre-processing
df_stats = pd.read_csv('data/player_per_game.csv', index_col=0).reset_index(drop=True)
# Launch the Dasboard in streamlit
st.title('Forecast model for NBA players')
st.write('Linear regression model to predict the players BPM (Box-Plus-Minus)')

st.header('Data pre-processing')
# Graph with no. minutes played
hist_fig = px.histogram(df_stats, x='mp', nbins=30, title='Histogram of minutes played', template='plotly_white')
st.write(hist_fig)