## File with all functions used in the application
# Import packages
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import linear_model # Linear regression
from sklearn import svm
from sklearn import preprocessing

#%% Functions
def df_feat_describe(data, cols_name):
    return data.describe()[cols_name].transpose().reset_index().rename({'index': "var"}, axis=1)

def feat_bar_plot(data, plt_title, log_scale):
    return px.bar(data[['var', 'mean', 'std']].melt(id_vars=['var']),
                  x='var', y='value', color='variable', barmode='group',
                  title=plt_title,
                  template='plotly_white', labels={"var": "Variable", "value": "Value", "variable": "Statistic"},
                  color_discrete_sequence=px.colors.qualitative.Safe, log_y=log_scale,)

def reg_model_selection(name):
    if name == 'Stochastic Gradient Descent':
        return linear_model.SGDRegressor(loss='squared_loss', penalty='l2', max_iter=1000)
    elif name == 'Ridge Regression':
        return linear_model.Ridge(alpha=0.5)
    elif name == 'Support Vector Regression':
        return svm.SVR(kernel='rbf', degree=3)

def normalize_dataset(df):
    # Scale the DF using sktLearn-preprocessing
    scaler = preprocessing.StandardScaler().fit(df)  # We use a standard scale (mean = 0, std = 1)
    # Fit(cont_df) to that scaler
    return scaler.transform(df)  # Array