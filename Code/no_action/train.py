import numpy as np
import os, pickle

import src.helper as helper
import src.train_functions as train_functions
import no_action.conf as conf

def train(model, folder, ship_model_folder):

    files = os.listdir(folder)
    files.sort(key=helper.natural_keys)
    nbr_games = len(files)

    if nbr_games < 101:
        files = files[:-1]
        nbr_train_games = nbr_games-1
    else:
        nbr_train_games = 100
        files = files[-101:-1]

    print('Training on {} games'.format(nbr_train_games))

    conf.get('train_function')(model, folder, files, conf.get('input_data'))
    model.save(ship_model_folder, save_format='tf')