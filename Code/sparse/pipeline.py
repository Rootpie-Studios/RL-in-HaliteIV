import tensorflow as tf
import time, random

import src.helper as helper
import src.model as model
import src.pipeline_helper as pipeline_helper
import sparse.train as train
import sparse.conf as conf

with tf.device('/CPU:0'):
    try:
        ship_agent = tf.keras.models.load_model(conf.get('SHIP_MODEL'))
    except:
        ship_agent = conf.get('build_model')(nbr_imgs=9)
        ship_agent.save(conf.get('SHIP_MODEL'), save_format='tf')

    enemies = [ 'agents/baseline500.py', 
                'agents/baseline536.py', 
                'agents/bigger_nn_607.py', 
                'agents/bigger_nn_sparse116.py', 
                'agents/bigger_nn_sparse150.py', 
                'agents/bigger_nn_sparse280.py',
                'agents/enemy_data539.py',
                'agents/gamma01.py',
                'agents/gamma05.py',
                'agents/nearsighted.py',
                'agents/no_action.py',
                'agents/p26_5.py',
                'agents/p26_10.py',
                'agents/p26_15.py',
                'agents/p26.py',
                'agents/small_deep_nn.py',
                'agents/smaller_nn_551.py',
                'random']

    pipeline_helper.pipeline_fnc_sparse(ship_agent, 
        conf.get('SHIP_MODEL'), 
        conf.get('AGENT1'), 
        conf.get('AGENT1_EXPLOIT'), 
        enemies, 
        conf.get('SESSIONS'), 
        conf.get('GAMES'), 
        {'size':conf.get('size'), 'episodeSteps':conf.get('episodeSteps')}, 
        conf.get('PLAY_EXPLOIT'),
        conf.get('GAMES_FOLDER'),
        conf.get('HTML_FOLDER'),
        conf.get('EXPLOIT_GAMES_FOLDER'), 
        conf.get('EXPLOIT_HTML_FOLDER'),
        train.train)