## Linear regression method for NBA players
# We use historic data to predict the BPM (Box-Plus-Minus)
# BPM is a basketball box score-based metric that estimates a player’s contribution to the team when he/she is on the court.
# BPM uses a player’s box score information, position, and the team’s overall performance to estimate the player’s contribution in points above league average per 100 possessions played.
# A linear regression model is implemented to predict the BPM of NBA players.
# We build a dashboard in Streamlit to show the results.

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
data_dir = '~/PycharmProjects/Regression_models_in_Sports/data/'
file_name='player_per_game.csv'
df_stats = pd.read_csv(str(data_dir+file_name), index_col=0).reset_index(drop=True)
# Launch the Dasboard in streamlit
st.title('Forecast model for NBA players')
st.write('Linear regression model to predict the players BPM (Box-Plus-Minus)')

st.header('Data pre-processing')
# Graph with no. minutes played
hist_fig = px.histogram(df_stats, x='mp', nbins=30, title='Histogram of minutes played', template='plotly_white')
st.write(hist_fig)
# Filter out the small samples
df_stats = df_stats[df_stats['mp'] >= 500].reset_index(drop=True)
# Correlations of features - 2D evaluation
st.subheader('Correlations of features')
corr_x = st.selectbox('Correlation - X axis:', options=df_stats.columns, index=df_stats.columns.get_loc('ts_pct'))
#corr_y = st.selectbox('Correlation - Y axis:', options=df_stats.columns, index=df_stats.columns.get_loc('mp_per_g'))
corr_y = st.selectbox('Correlation - Y axis:', options=['bpm', 'per'], index=0)
# Differentiate the scatter-points by color
corr_col = st.radio('Correlation - color axis:', options=['age', 'season'], index=0)
corr_fig = px.scatter(df_stats, x=corr_x, y=corr_y, title=f'Correlation between {corr_x} and {corr_y}',
                 template='plotly_white', render_mode='webgl', color=corr_col,
                 hover_data=['name', 'pos', 'age', 'season'], color_continuous_scale=px.colors.sequential.OrRd
)
corr_fig.update_traces(mode='markers', marker={'line': {'width': 0.4, 'color': 'slategrey'}}) # Define the color scheme
st.write(corr_fig)

# Filter out the necessary features
cont_var_cols = ['g', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg_pct', 'fg3_per_g', 'fg3a_per_g', 'fg3_pct', 'fg2_per_g', 'fg2a_per_g',
                 'fg2_pct', 'efg_pct', 'ft_per_g', 'fta_per_g', 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g',
                 'tov_per_g', 'pf_per_g', 'pts_per_g']
cont_df = df_stats[cont_var_cols] # Only keep relevant features

