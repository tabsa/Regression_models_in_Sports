## Docker server for the regression model application
# File that Docker server will run, which is the main file that calls the regression model application

#%% Import packages and python.files
from pathlib import Path
from func import *
from reg_model import reg_mdl
from app import dashboard

#%% Main file
if __name__ == '__main__':
    data_dir = Path.cwd()
    file_name='player_per_game.csv'
    # Filter out the necessary features
    cont_var_cols = ['g', 'mp_per_g', 'fg_per_g', 'fga_per_g', 'fg3_per_g', 'fg3a_per_g', 'fg2_per_g','fg2a_per_g',
                     'efg_pct', 'ft_per_g', 'fta_per_g','orb_per_g', 'drb_per_g', 'trb_per_g', 'ast_per_g', 'stl_per_g',
                     'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g', 'mp']  # No column [bpm ; per]

    app_cls = dashboard(data_dir, file_name, cont_var_cols)

    #app_cls.run_dashboard() # Run the app with dashboard
    app_cls.run_script()
