## Dashboard of the regression models for NBA players

#%% Import packages
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

#%% Import python.files
from func import *
from reg_model import reg_mdl

#%% Dashboard class
class dashboard:
    def __init__(self, dir, file, feat_cols):
        self.path = str(dir + file)
        self.df_stats = self.get_data()
        self.feat_cols = feat_cols # Filter out the necessary features from df_stats
        self.cont_df = self.df_stats[self.feat_cols] # Repeated operation as in reg_mdl - Get DF with all features
        self.mdl_names = ["Stochastic Gradient Descent", "Ridge Regression", "Support Vector Regression"]  # Future it can be added others

    #@classmethod
    @st.cache()  # Memory catche to avoid multiple loads of the same data
    def get_data(self):
        return pd.read_csv(self.path, index_col=0).reset_index(drop=True)

    @st.cache()
    def get_reg_mdl(self, y_opt, models):
        reg_class = reg_mdl(self.df_stats, self.feat_cols, y_opt, models)
        reg_class.run_reg_models()
        return reg_class

    def run_dashboard(self):
        self.head_title()
        self.sec_preprocessing()
        self.sec_exploration()
        reg_opt = self.sec_reg_model()
        self.sec_eval_results(reg_opt)

    def run_script(self):
        # Select BPM (Box-Plus-Minus) or PER (Player Efficiency Rating) to be predicted by the model
        y_stat = 'bpm'
        # Call the call reg_models to build and run the models
        mdl_names = ["Stochastic Gradient Descent", "Ridge Regression",
                     "Support Vector Regression"]  # Future it can be added others
        self.reg_models_class = self.get_reg_mdl(y_stat, mdl_names)
        print('Simulation done')
        print(self.reg_models_class.df_test)

    def head_title(self):
        # Launch the Dasboard in streamlit
        st.title('Forecast model for NBA players')
        st.write('Linear regression model to predict the players BPM (Box-Plus-Minus) and PER (Player Efficiency Rating)')

    def sec_preprocessing(self):
        st.header('Data pre-processing')
        if st.checkbox('Show DataFrame with raw data'):
            st.write(self.df_stats)
        # Graph with no. minutes played
        hist_fig = px.histogram(self.df_stats, x='mp', nbins=30, title='Histogram of minutes played', template='plotly_white')
        st.write(hist_fig)
        # Filter out the small samples
        self.df_stats = self.df_stats[self.df_stats['mp'] >= 500].reset_index(drop=True)
        # Correlations of features - 2D evaluation
        st.subheader('Correlations of features')
        corr_x = st.selectbox('Correlation - X axis:', options=self.df_stats.columns, index=self.df_stats.columns.get_loc('pts_per_g'))
        # corr_y = st.selectbox('Correlation - Y axis:', options=df_stats.columns, index=df_stats.columns.get_loc('mp_per_g'))
        corr_y = st.selectbox('Correlation - Y axis:', options=['bpm', 'per'], index=0)
        # Differentiate the scatter-points by color
        corr_col = st.radio('Correlation - color axis:', options=['age', 'season'], index=0)
        corr_fig = px.scatter(self.df_stats, x=corr_x, y=corr_y, title=f'Correlation between {corr_x} and {corr_y}',
                              template='plotly_white', render_mode='webgl', color=corr_col,
                              hover_data=['name', 'pos', 'age', 'season'],
                              color_continuous_scale=px.colors.sequential.OrRd
                              )
        corr_fig.update_traces(mode='markers',
                               marker={'line': {'width': 0.4, 'color': 'slategrey'}})  # Define the color scheme
        st.write(corr_fig)

    def sec_exploration(self):
        st.header('Data exploration')
        st.write('Check the graph below for the Correlation between [pts_per_g; bpm] and [pf_per_g; bpm]:')
        st.markdown("""
                >There is some features with correlation with bpm (pts_per_g, ast_per_g, etc) and others with non-correlation (pf_per_g)
                """)
        # Pandas profiling
        feat_desc = df_feat_describe(self.cont_df, self.feat_cols)
        # [].transpose().reset_index is to transpose the features as index and reset them from 0,1,...,N
        if st.checkbox('Show feature_DataFrame'):
            st.write(feat_desc)
        st.subheader('Feature scaling plot')
        plot_scale = st.radio('Plot scale:', options=['normal', 'log'], index=0)
        log_scale = False  # Default option (plot_scale = normal)
        if plot_scale == 'log':
            log_scale = True
        # Bar plot
        feat_fig = feat_bar_plot(feat_desc, f'Statistical description of various features — plot with {plot_scale} scale',
                                 log_scale)
        st.write(feat_fig)
        st.write("We have quite discrepancy of various features - it requires normalizing to a scale [0-1]")
        # Calling the func normalize_dataset (scale the DF using sktLearn-preprocessing)
        X = normalize_dataset(self.cont_df)
        self.cont_df_scaler = pd.DataFrame(X, columns=self.feat_cols)  # DataFrame
        if st.checkbox('Show normalized DataFrame'):
            st.write(self.cont_df_scaler)
        # Execute the same operation
        feat_desc = df_feat_describe(self.cont_df_scaler, self.feat_cols)
        feat_fig = feat_bar_plot(feat_desc, f'Statistical description of various features with Standard scaling', False)
        st.write(feat_fig)

    def sec_reg_model(self):
        st.header('Prediction results')
        # Select BPM (Box-Plus-Minus) or PER (Player Efficiency Rating) to be predicted by the model
        y_stat = st.selectbox('Select BPM or PER to be predicted:', ['bpm', 'per'], index=0)
        # Select the regression model: Stochastic grad desc, Ridge regression, Supp vector regression
        reg_opt = st.multiselect("Choose regression model", options=self.mdl_names,
                                 default=self.mdl_names[0])  # Returns list with all selected models
        # Call the call reg_models to build and run the models
        self.reg_models_class = self.get_reg_mdl(y_stat, self.mdl_names)
        # Plot of Y_pred vs Y_test
        val_fig = px.scatter(self.reg_models_class.df_test, x="Actual", y=reg_opt,
                             title=f"Prediction of {y_stat.upper()} vs ground truths", template="plotly_white",
                             color_discrete_sequence=px.colors.qualitative.Safe, hover_data=["player", "season"]
                             )
        st.write(val_fig)
        return reg_opt

    def sec_eval_results(self, reg_opt):
        st.subheader("Evaluation of the results:")
        st.write("Mean square error:")
        st.write(self.reg_models_class.df_mse[reg_opt])