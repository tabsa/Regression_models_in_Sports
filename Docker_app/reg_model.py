## Linear regression method for NBA players
# We use historic data to predict the BPM (Box-Plus-Minus)
# BPM is a basketball box score-based metric that estimates a player’s contribution to the team when he/she is on the court.
# BPM uses a player’s box score information, position, and the team’s overall performance to estimate the player’s contribution in points above league average per 100 possessions played.
# A linear regression model is implemented to predict the BPM of NBA players.
# We build a dashboard in Streamlit to show the results.

#%% Import packages
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
# Import functions
from func import reg_model_selection

#%% Build the Regression model
class reg_mdl:
    def __init__(self, df, x_feat_opt, y_opt, mdl_list, norm_opt=True):
        """
        :param df: Original DataFrame - df_stats (pd.DataFrame)
        :param x_feat_opt: Name of all df_columns - cont_var_cols (String)
        :param y_opt: Name of df_column for prediction (BPM or PER)
        :param mdl_list: List with name of the methods - st.multiselect from the dashboard
        """
        self.y_opt = y_opt
        self.x_feat_opt = x_feat_opt
        self.df = df
        self.cont_df = df[x_feat_opt]
        self.reg_opt = mdl_list
        self.norm_opt = norm_opt # Normalization option (default = True)

    def normalize_dataset(self):
        # Scale the DF using sktLearn-preprocessing
        scaler = preprocessing.StandardScaler().fit(self.cont_df)  # We use a standard scale (mean = 0, std = 1)
        # Fit(cont_df) to that scaler
        return scaler.transform(self.cont_df)  # Array
        #self.cont_df_scaler = pd.DataFrame(self.X, columns=self.x_feat_opt)  # DataFrame
        # It goes to the dashboard class

    def split_train_test(self):
        # Normalize the datastet
        self.X = self.normalize_dataset() if self.norm_opt==True else self.cont_df
        # Select the only target for prediction - BPM (Box-Plus-Minus) or PER (Player Efficiency Rating)
        self.Y = self.df[self.y_opt].values
        # Split the data into training and testing
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(self.X, self.Y, train_size=0.8, random_state=42, shuffle=True)
        _, df_test_name = model_selection.train_test_split(self.df, train_size=0.8, random_state=42, shuffle=True) # Get individual info of each Y_test element - Name, position, pts_per_g, season, etc
        return X_train, X_test, Y_train, Y_test, df_test_name

    def run_reg_models(self):
        # Split the dataset into train and test
        X_train, X_test, Y_train, Y_test, df_test_name = self.split_train_test()
        model = []
        Y_test_hat = np.zeros((Y_test.shape[0], len(self.reg_opt))) # np.array for predicted values per model
        mse = np.zeros(len(self.reg_opt)) # MSE value per model
        for i in range(len(self.reg_opt)): # For-loop per selected model (default: Sto Grad Descent)
            model.append(reg_model_selection(self.reg_opt[i])) # Creating model
            model[-1].fit(X_train, Y_train) # Training the model
            Y_test_hat[:, i] = model[-1].predict(X_test) # Predict with Y_test
            mse[i] = metrics.mean_squared_error(Y_test, Y_test_hat[:, i]) # Calculate the MSE value

        # Outcome: Dataframe: [Y_pred, Y_test]
        self.Y_res = np.column_stack((Y_test_hat, Y_test))
        self.reg_opt.append('Actual') # Add another column name
        self.df_test = pd.DataFrame(self.Y_res.T, index=self.reg_opt).transpose()
        # Add 2-extra columns with players Name and Season
        self.df_test = self.df_test.assign(player=df_test_name["name"].values)
        self.df_test = self.df_test.assign(season=df_test_name["season"].values)

        # Outcome: Dataframe: [mse]
        self.df_mse = pd.DataFrame([mse], columns=self.reg_opt[:-1])
