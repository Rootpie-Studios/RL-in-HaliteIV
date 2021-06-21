from kaggle_environments.envs.halite.helpers import *
from kaggle_environments import make
import numpy as np
import tensorflow as tf
import math, os, pickle, time, re

import se_sm_de_kp_ez_te.conf as conf

with tf.device('/CPU:0'):
    agent1 = 'se_sm_de_kp_ez_te/exploit_no_save.py'
    agent2 = 'se_sm_de_kp_ez_te/p26.py'
    env = make('halite', debug=True, configuration={'size':conf.get('size'), 'episodeSteps':conf.get('episodeSteps')})
    obs = env.run([agent1, agent2])

    # HTML saving
    with open('se_sm_de_kp_ez_te/test_html' + '/' + agent1[23:-3] + '_vs_' + agent2[23:-3] + '.html', 'wt') as fd:
        fd.write(env.render(mode='html'))

    print('Game saved\n###########')