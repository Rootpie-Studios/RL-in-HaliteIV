import tensorflow as tf
import time

import src.helper as helper
import src.model as model
import src.pipeline_helper as pipeline_helper
import eps_zero.train as train
import eps_zero.conf as conf

with tf.device('/CPU:0'):
    try:
        ship_agent = tf.keras.models.load_model(conf.get('SHIP_MODEL'))
    except:
        ship_agent = conf.get('build_model')()
        ship_agent.save(conf.get('SHIP_MODEL'), save_format='tf')

    pipeline_helper.pipeline_fnc(ship_agent, 
        conf.get('SHIP_MODEL'), 
        conf.get('AGENT1'), 
        conf.get('AGENT1_EXPLOIT'), 
        conf.get('AGENT2'), 
        conf.get('SESSIONS'), 
        conf.get('GAMES'), 
        {'size':conf.get('size'), 'episodeSteps':conf.get('episodeSteps')}, 
        conf.get('PLAY_EXPLOIT'),
        conf.get('GAMES_FOLDER'),
        conf.get('HTML_FOLDER'),
        conf.get('EXPLOIT_GAMES_FOLDER'), 
        conf.get('EXPLOIT_HTML_FOLDER'),
        train.train)
