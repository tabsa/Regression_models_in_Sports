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

#%% Functions
def df_feat_describe(data, cols_name):
    return data.describe()[cols_name].transpose().reset_index().rename({'index': "var"}, axis=1)

def feat_bar_plot(data, plt_title, log_scale):
    return px.bar(data[['var', 'mean', 'std']].melt(id_vars=['var']),
                  x='var', y='value', color='variable', barmode='group',
                  title=plt_title,
                  template='plotly_white', labels={"var": "Variable", "value": "Value", "variable": "Statistic"},
                  color_discrete_sequence=px.colors.qualitative.Safe, log_y=log_scale,)

#%% Data pre-processing
data_dir = '~/PycharmProjects/Regression_models_in_Sports/data/'
file_name='player_per_game.csv'
df_stats = pd.read_csv(str(data_dir+file_name), index_col=0).reset_index(drop=True)
# Launch the Dasboard in streamlit
st.title('Forecast model for NBA players')
st.write('Linear regression model to predict the players BPM (Box-Plus-Minus) and PER (Player Efficiency Rating)')

st.header('Data pre-processing')
# Graph with no. minutes played
hist_fig = px.histogram(df_stats, x='mp', nbins=30, title='Histogram of minutes played', template='plotly_white')
st.write(hist_fig)
# Filter out the small samples
df_stats = df_stats[df_stats['mp'] >= 500].reset_index(drop=True)
# Correlations of features - 2D evaluation
st.subheader('Correlations of features')
corr_x = st.selectbox('Correlation - X axis:', options=df_stats.columns, index=df_stats.columns.get_loc('pts_per_g'))
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
cont_var_cols = ['g', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g', 'fg2a_per_g', 'efg_pct', 'ft_per_g', 'fta_per_g',
                 'orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp']
cont_df = df_stats[cont_var_cols] # Only keep relevant features

#%% Data exploration
st.header('Data exploration')
st.write('Check the graph below for the Correlation between [pts_per_g; bpm] and [pf_per_g; bpm]:')
st.markdown("""
        >There is some features with correlation with bpm (pts_per_g, ast_per_g, etc) and others with non-correlation (pf_per_g)
        """)
# Pandas profiling
feat_desc = df_feat_describe(cont_df, cont_var_cols)
# [].transpose().reset_index is to transpose the features as index and reset them from 0,1,...,N
if st.checkbox('Show feature_DataFrame'):
    st.write(feat_desc)
st.subheader('Feature scaling plot')
plot_scale = st.radio('Plot scale:', options=['normal', 'log'], index=0)
log_scale=False # Default option (plot_scale = normal)
if plot_scale == 'log':
    log_scale=True
# Bar plot
feat_fig = feat_bar_plot(feat_desc, f'Statistical description of various features — plot with {plot_scale} scale', log_scale)
st.write(feat_fig)
st.write("We have quite discrepancy of various features - it requires normalizing to a scale [0-1]")
# Scale the DF using sktLearn-preprocessing
scaler = preprocessing.StandardScaler().fit(cont_df) # We use a standard scale (mean = 0, std = 1)
# Fit(cont_df) to that scaler
X = scaler.transform(cont_df) # Array
cont_df_scaler = pd.DataFrame(X, columns=cont_var_cols) # DataFrame
if st.checkbox('Show normalized DataFrame'):
    st.write(cont_df_scaler)
# Execute the same operation
feat_desc = df_feat_describe(cont_df_scaler, cont_var_cols)
feat_fig = feat_bar_plot(feat_desc, f'Statistical description of various features with Standard scaling', False)
st.write(feat_fig)

#%% Build the Regression model
# Select BPM (Box-Plus-Minus) or PER (Player Efficiency Rating) to be predicted by the model
y_stat = st.selectbox('Select BPM or PER to predicts:', ['bpm', 'per'], index=0)
Y = df_stats[y_stat].values # Select the only target for prediction
# Split the data into training and testing
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=42, shuffle=True)