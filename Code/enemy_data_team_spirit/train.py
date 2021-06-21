import numpy as np
import os, pickle

import src.helper as helper
import src.train_functions as train_functions
import enemy_data_team_spirit.conf as conf

def train(model, folder, ship_model_folder):

    files = os.listdir(folder)
    files.sort(key=helper.natural_keys)
    nbr_games = len(files)

    if nbr_games < 11:
        files = files[:-1]
        nbr_train_games = nbr_games-1
    else:
        nbr_train_games = 10
        files = files[-11:-1]

    print('Training on {} games'.format(nbr_train_games))

    conf.get('train_function')(model, folder, files, conf.get('input_data'), reward_fnc=conf.get('reward_function'), sample_rate=1)
    model.save(ship_model_folder, save_format='tf')