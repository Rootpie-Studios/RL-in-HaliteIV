from kaggle_environments.envs.halite.helpers import *

import numpy as np
import tensorflow as tf
import pickle, collections, random, os

import src.model as model
import src.helper as helper
import src.agent_helper as agent_helper
import src.input_data as input_data
import gamma05.conf as conf

# Initiate memory for game
gameID = str(len(os.listdir(conf.get('GAMES_FOLDER'))))
print('Starting game nbr:', gameID)

memory = {'game' : collections.deque(maxlen=401), 'reward' : 0}
with open(conf.get('GAMES_FOLDER') + '/' + gameID, 'wb') as pickle_file:
    pickle.dump(memory, pickle_file)

try:
    ship_agent = tf.keras.models.load_model(conf.get('SHIP_MODEL'))
except:
    ship_agent = conf.get('build_model')()
    ship_agent.save(conf.get('SHIP_MODEL'), save_format='tf')


def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player

    own_ship_actions = {}
    all_other_ship_actions = {}

    new_ship_pos = []

    sorted_ships = sorted(me.ships, key=lambda x: x.halite, reverse=True)
    shipyard_build = False

    # Calculate epsilon for exploration rate
    if conf.get('epsilon_decay'):
        epsilon = conf.get('epsilon')-conf.get('epsilon_decay_rate')*len(os.listdir(conf.get('GAMES_FOLDER')))
        epsilon = max(conf.get('min_epsilon'), epsilon)
    else:
        epsilon = conf.get('epsilon')
        
    r_per_ship = 0
    r_all_ships = 0
    # Ships
    for ship in sorted_ships:
        state, has_nearby_shipyards = conf.get('input_data')(obs, ship._id, me._id, own_ship_actions, conf.get('size'))

        action, Qs = agent_helper.get_action(epsilon, ship_agent, state)

        r = float(Qs[action])
        r_all_ships += r

        if action != 0:
            ship.next_action = helper.ship_action_string(action)
        
        # Convert strat
        action, shipyard_build = agent_helper.convert_strat(me, ship, has_nearby_shipyards, action, shipyard_build)

        # Updating ship map for next ship
        ship_pos = helper.get_next_ship_pos(obs['players'][me._id][2][ship._id][0], action)
        new_ship_pos.append(ship_pos)
        all_other_ship_actions.update({ship._id : own_ship_actions.copy()})
        own_ship_actions.update({ship._id : action})
    
    # Correcting reward for amount of ships
    if len(sorted_ships) > 0:
        r_per_ship = r_all_ships/len(sorted_ships)

    # Convert override
    agent_helper.convert_override(me, sorted_ships)

    # Shipyard strat
    agent_helper.shipyard_strat(me, obs, board, new_ship_pos, conf.get('episodeSteps'))

    # Save state and action to memory
    with open(conf.get('GAMES_FOLDER') + '/' + gameID, 'rb') as pickle_file:
        memory = pickle.load(pickle_file)
    memory['game'].append((obs, me.next_actions, all_other_ship_actions))
    memory['reward'] += r_per_ship

    with open(conf.get('GAMES_FOLDER') + '/' + gameID, 'wb') as pickle_file:
        pickle.dump(memory, pickle_file)

    return me.next_actions