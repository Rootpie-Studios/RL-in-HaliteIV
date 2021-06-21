import numpy as np
import os, pickle

import src.helper as helper
import sparse.conf as conf

def train(model, folder, ship_model_folder):

    files = os.listdir(folder)
    files.sort(key=helper.natural_keys)
    nbr_games = len(files)

    if nbr_games < 31:
        files = files[:-1]
        nbr_train_games = nbr_games-1
    else:
        nbr_train_games = 30
        files = files[-31:-1]

    print('Training on {} games'.format(nbr_train_games))

    conf.get('train_function')(model, folder, files, conf.get('input_data'), nbr_imgs=9, gamma=0.96, sample_rate=1)
    model.save(ship_model_folder, save_format='tf')