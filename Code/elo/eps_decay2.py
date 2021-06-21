from kaggle_environments.envs.halite.helpers import *

import numpy as np
import tensorflow as tf
import pickle, collections, random, os
import helper, conf, input_data

ship_model = tf.keras.models.load_model('models/eps_decay2_model')

def agent(obs, config):
    board = Board(obs, config)
    me = board.current_player

    own_ship_actions = {}
    all_other_ship_actions = {}

    new_ship_pos = []

    sorted_ships = sorted(me.ships, key=lambda x: x.halite, reverse=True)
    shipyard_build = False

    # Ships
    for ship in sorted_ships:
        state, has_nearby_shipyards = input_data.obs_to_matrix_baseline(obs, ship._id, me._id, own_ship_actions)

        Qs = ship_model(state, training=False)[0]
        action = np.argmax(Qs)

        if action != 0:
            ship.next_action = helper.ship_action_string(action)
        
        # Convert strategy
        if me.halite + ship.halite > 500 and not has_nearby_shipyards and not shipyard_build:
            ship.next_action = ShipAction.CONVERT
            action = 5
            shipyard_build = True

        # Updating ship map for next ship
        ship_pos = helper.get_next_ship_pos(obs['players'][me._id][2][ship._id][0], action)
        new_ship_pos.append(ship_pos)
        all_other_ship_actions.update({ship._id : own_ship_actions.copy()})
        own_ship_actions.update({ship._id : action})

    # Convert override
    if len(me.shipyards) == 0 and len(me.ships) > 0:
        ship = sorted_ships[0]
        if ship.halite + me.halite > 500 and len(me.ships) > 1:
            ship.next_action = ShipAction.CONVERT
        elif ship.halite + me.halite > 1000:
            ship.next_action = ShipAction.CONVERT

    # Shipyards
    if len(me.shipyards) > 0:
        shipyard = random.choice(list(me.shipyards))
        ship_nearby = False

        if obs['players'][me._id][1][shipyard._id] in new_ship_pos:
            ship_nearby = True

        if me._halite >= 500 and not ship_nearby and board.step < conf.get('episodeSteps') - 100 and len(me.ships) < 20:
            shipyard.next_action = ShipyardAction.SPAWN
        elif me._halite >= 500 and not ship_nearby and len(me.ships) < 10 and board.step < conf.get('episodeSteps') - 10:
            shipyard.next_action = ShipyardAction.SPAWN

    return me.next_actions