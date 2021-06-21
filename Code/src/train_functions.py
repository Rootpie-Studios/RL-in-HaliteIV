import numpy as np
import os, pickle

import src.input_data as input_data
import src.helper as helper
import src.reward as reward

def train_on_games_baseline(model, folder, files, input_data_fnc, gamma=0.9, img_size=21, nbr_imgs=8, action_space=5, epochs=5, batch_size=32, sample_rate=0.1, size=21, reward_fnc=reward.halite_reward):
    inputs_data = []
    inputs_image = []
    targets = []
    player_id = 0

    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        game = game_data['game']
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0
        won = 5000 if winning_player == 0 else -100

        for index, (obs, actions, other_ship_actions) in enumerate(game):
            ships = obs['players'][0][2]
            shipyards = obs['players'][0][1]
            # Looping over ships
            for ship_id, ship_info in ships.items():
                last_frame = False
                # Only prepare sample_rate of samples
                if sample_rate > np.random.rand():
                    # If not end of game and ship in next_ships
                    if index < len(game) - 1:
                        next_ships = game[index + 1][0]['players'][0][2].keys()
                        if ship_id in next_ships:
                            ns = True
                            next_obs = game[index + 1][0]
                            own_ship_actions = game[index + 1][2][ship_id]
                            next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                        else:
                            ns = False
                    else:
                        last_frame = True
                        ns = False

                    # Updating moved ships
                    what_does_our_ships = other_ship_actions[ship_id]
                    next_ship_poses = helper.get_next_ship_poses(ships, what_does_our_ships, ship_id)

                    if ship_id in actions:
                        ship_action = helper.ship_action_index(actions[ship_id])
                    else:
                        ship_action = 0

                    next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)
                    danger_poses = helper.get_danger_poses(obs, ship_info)
                    
                    # Getting state
                    state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)

                    # Target and reward calculations
                    target = np.array(model(state, training=False)[0])
                    if ship_action != 5:
                        if ns:
                            r, os = reward_fnc(ns, ship_action, ships[ship_id][1], next_obs['players'][0][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            if not os: 
                                target[ship_action] = r + np.amax(model(next_state, training=False)[0]) * gamma
                        else:
                            r, os = reward_fnc(ns, ship_action, ships[ship_id][1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            if not os:
                                target[ship_action] = r

                    # inputs_data.append(state[0])
                    inputs_image.append(state)
                    targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        p = np.random.permutation(len(nn_inputs_image))
        nn_inputs_image = nn_inputs_image[p]
        nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size)

def train_on_games_final(model, folder, files, input_data_fnc, gamma=0.9, img_size=21, nbr_imgs=9, action_space=5, epochs=5, batch_size=32, sample_rate=0.1, size=21, reward_fnc=reward.final_reward):
    inputs_data = []
    inputs_image = []
    targets = []
    player_id = 0

    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        game = game_data['game']
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0
        won = 5000 if winning_player == 0 else -100

        for index, (obs, actions, other_ship_actions) in enumerate(game):
            ships = obs['players'][0][2]
            shipyards = obs['players'][0][1]
            frame_target_action_pairs = []
            frame_states = []
            frame_reward = 0
            next_frame_reward_sum = 0
            # Looping over ships
            for ship_id, ship_info in ships.items():
                last_frame = False
                # Only prepare sample_rate of samples
                if sample_rate > np.random.rand():
                    # If not end of game and ship in next_ships
                    if index < len(game) - 1:
                        next_ships = game[index + 1][0]['players'][0][2].keys()
                        if ship_id in next_ships:
                            ns = True
                            next_obs = game[index + 1][0]
                            own_ship_actions = game[index + 1][2][ship_id]
                            next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                        else:
                            ns = False
                    else:
                        last_frame = True
                        ns = False

                    # Updating moved ships
                    what_does_our_ships = other_ship_actions[ship_id]
                    next_ship_poses = helper.get_next_ship_poses(ships, what_does_our_ships, ship_id)

                    if ship_id in actions:
                        ship_action = helper.ship_action_index(actions[ship_id])
                    else:
                        ship_action = 0

                    next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)
                    danger_poses = helper.get_danger_poses(obs, ship_info)
                    
                    # Getting state
                    state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)

                    # Target and reward calculations
                    target = np.array(model(state, training=False)[0])
                    if ship_action != 5:
                        future_reward = 0
                        if ns:
                            r = reward_fnc(ns, ship_action, ships[ship_id][1], next_obs['players'][0][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            frame_reward += r 
                            future_reward = np.amax(model(next_state, training=False)[0]) * gamma
                            next_frame_reward_sum += future_reward
                            own_reward = r
                        else:
                            r = reward_fnc(ns, ship_action, ships[ship_id][1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            frame_reward += r
                            own_reward = r

                        if last_frame:
                            potential_kills = helper.get_kill_poses(obs, ship_info, 1)

                            if len(game) == 399:
                                frame_reward += won
                            elif next_ship_pos in potential_kills:
                                frame_reward += won

                        frame_states.append(state)
                        frame_target_action_pairs.append((target, ship_action, own_reward, future_reward))
                    # inputs_data.append(state[0])
                    # inputs_image.append(state)
                    # targets.append(target)
                if len(ships) > 0:
                    next_frame_reward_average = next_frame_reward_sum / len(ships)

                    for state, (target, action, own_reward, future_reward) in zip(frame_states, frame_target_action_pairs):
                        target[action] = 0.5 * own_reward + 0.5 * (frame_reward/50) + (future_reward*0.5 + next_frame_reward_average*0.5)

                        inputs_image.append(state)
                        targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        p = np.random.permutation(len(nn_inputs_image))
        nn_inputs_image = nn_inputs_image[p]
        nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size, validation_split=0.33)

def train_on_games_team_spirit(model, folder, files, input_data_fnc, gamma=0.9, img_size=21, nbr_imgs=8, action_space=5, epochs=5, batch_size=32, sample_rate=0.1, size=21, reward_fnc=reward.halite_reward):
    inputs_data = []
    inputs_image = []
    targets = []
    player_id = 0

    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        game = game_data['game']
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0
        won = 5000 if winning_player == 0 else -100

        for index, (obs, actions, other_ship_actions) in enumerate(game):
            # Only prepare sample_rate of samples
            if sample_rate > np.random.rand():
                ships = obs['players'][0][2]
                shipyards = obs['players'][0][1]
                frame_target_action_pairs = []
                frame_states = []
                frame_reward = 0
                next_frame_reward_sum = 0
                # Looping over ships
                for ship_id, ship_info in ships.items():
                    last_frame = False
                    # If not end of game and ship in next_ships
                    if index < len(game) - 1:
                        next_ships = game[index + 1][0]['players'][0][2].keys()
                        if ship_id in next_ships:
                            ns = True
                            next_obs = game[index + 1][0]
                            own_ship_actions = game[index + 1][2][ship_id]
                            next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                        else:
                            ns = False
                    else:
                        last_frame = True
                        ns = False

                    # Updating moved ships
                    what_does_our_ships = other_ship_actions[ship_id]
                    next_ship_poses = helper.get_next_ship_poses(ships, what_does_our_ships, ship_id)

                    if ship_id in actions:
                        ship_action = helper.ship_action_index(actions[ship_id])
                    else:
                        ship_action = 0

                    next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)
                    danger_poses = helper.get_danger_poses(obs, ship_info)
                    
                    # Getting state
                    state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)
                    frame_states.append(state)

                    # Target and reward calculations
                    target = np.array(model(state, training=False)[0])
                    frame_target_action_pairs.append((target, ship_action))
                    if ship_action != 5:
                        if ns:
                            r, os = reward_fnc(ns, ship_action, ships[ship_id][1], next_obs['players'][0][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            if not os: 
                                next_frame_reward_sum += np.amax(model(next_state, training=False)[0]) * gamma
                                frame_reward += r 
                            else:
                                next_frame_reward_sum += np.amax(model(next_state, training=False)[0]) * gamma
                                frame_reward += target[ship_action]
                        else:
                            r, os = reward_fnc(ns, ship_action, ships[ship_id][1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            if not os:
                                frame_reward += r
                            else:
                                frame_reward += target[ship_action]

                if len(ships) > 0:
                    next_frame_reward_average = next_frame_reward_sum / len(ships)

                    for state, (target, action) in zip(frame_states, frame_target_action_pairs):
                        if action != 5:
                            target[action] = frame_reward + next_frame_reward_average

                            inputs_image.append(state)
                            targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        p = np.random.permutation(len(nn_inputs_image))
        nn_inputs_image = nn_inputs_image[p]
        nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size)

def train_on_games_enemy_data_team_spirit(model, folder, files, input_data_fnc, win_reward=10000, gamma=0.9, img_size=21, nbr_imgs=8, action_space=5, epochs=5, batch_size=32, sample_rate=1, size=21, reward_fnc=reward.halite_reward):
    inputs_data = []
    inputs_image = []
    targets = []
    player_id = 0

    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        data = [game_data['game'], game_data['enemy']]
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0

        for game in data:
            for index, (obs, actions, other_ship_actions, player_id) in enumerate(game):
                won = win_reward if player_id == 0 else 0
                
                enemy_id = 1 if player_id == 0 else 0
                # Only prepare sample_rate of samples
                if sample_rate > np.random.rand():
                    ships = obs['players'][player_id][2]
                    shipyards = obs['players'][player_id][1]
                    frame_target_action_pairs = []
                    frame_states = []
                    frame_reward = 0
                    next_frame_reward_sum = 0
                    ship_reward = {}
                    # Looping over ships
                    for ship_id, ship_info in ships.items():
                        last_frame = False
                        # If not end of game and ship in next_ships
                        if index < len(game) - 1:
                            next_ships = game[index + 1][0]['players'][player_id][2].keys()
                            if ship_id in next_ships:
                                ns = True
                                next_obs = game[index + 1][0]
                                own_ship_actions = game[index + 1][2][ship_id]
                                next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                            else:
                                ns = False
                        else:
                            last_frame = True
                            ns = False

                        # Updating moved ships
                        what_does_our_ships = other_ship_actions[ship_id]
                        next_ship_poses = helper.get_next_ship_poses(ships, what_does_our_ships, ship_id)

                        if ship_id in actions:
                            ship_action = helper.ship_action_index(actions[ship_id])
                        else:
                            ship_action = 0

                        next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)
                        danger_poses = helper.get_danger_poses(obs, ship_info,enemy_id=enemy_id)
                        
                        # Getting state
                        state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)
                        frame_states.append(state)

                        # Target and reward calculations
                        target = np.array(model(state, training=False)[0])
                        r = 0
                        if ship_action != 5:
                            if ns:
                                r = reward_fnc(ns, ship_action, ships[ship_id][1], next_obs['players'][player_id][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                                frame_reward += r
                                r += np.amax(model(next_state, training=False)[0]) * gamma
                            
                            else:
                                r = reward_fnc(ns, ship_action, ships[ship_id][1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                                frame_reward += r

                            if last_frame:
                                potential_kills = helper.get_kill_poses(obs, ship_info, 1)

                                if len(game) == 399 and ship_info[1] < 100:
                                    r += won
                                    frame_reward += won

                                elif next_ship_pos in potential_kills:
                                    r += won
                                    frame_reward += won

                        frame_target_action_pairs.append((target, ship_action, r))

                    if len(ships) > 0:
                        # next_frame_reward_average = next_frame_reward_sum / len(ships)

                        for state, (target, action, reward) in zip(frame_states, frame_target_action_pairs):
                            if action != 5:
                                target[action] = reward * 0.8 + frame_reward/10 * 0.2

                                inputs_image.append(state)
                                targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        # p = np.random.permutation(len(nn_inputs_image))
        # nn_inputs_image = nn_inputs_image[p]
        # nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size)


def train_on_games_and_enemy_data(model, folder, files, input_data_fnc, gamma=0.9, img_size=21, nbr_imgs=8, action_space=5, epochs=5, batch_size=32, sample_rate=0.1, size=21):
    inputs_data = []
    inputs_image = []
    targets = []
    
    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        data = [game_data['game'], game_data['enemy']]
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0
        won = 5000 if winning_player == 0 else -100

        for game in data:
            for index, (obs, actions, other_ship_actions, player_id) in enumerate(game):
                enemy_id = 1 if player_id == 0 else 0
                ships = obs['players'][player_id][2]
                shipyards = obs['players'][player_id][1]
                # Looping over ships
                for ship_id, ship_info in ships.items():
                    last_frame = False
                    # Only prepare sample_rate of samples
                    if sample_rate > np.random.rand():
                        # If not end of game and ship in next_ships
                        if index < len(game) - 1:
                            next_ships = game[index + 1][0]['players'][player_id][2].keys()
                            if ship_id in next_ships:
                                ns = True
                                next_obs = game[index + 1][0]
                                own_ship_actions = game[index + 1][2][ship_id]
                                next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                            else:
                                ns = False
                        else:
                            last_frame = True
                            ns = False

                        # Updating moved ships
                        what_does_our_ships = other_ship_actions[ship_id]
                        next_ship_poses = helper.get_next_ship_poses(ships, what_does_our_ships, ship_id)

                        if ship_id in actions:
                            ship_action = helper.ship_action_index(actions[ship_id])
                        else:
                            ship_action = 0

                        next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)
                        danger_poses = helper.get_danger_poses(obs, ship_info, enemy_id=enemy_id)
                        
                        # Getting state
                        state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)

                        # Target and reward calculations
                        target = np.array(model(state, training=False)[0])
                        if ship_action != 5:
                            if ns:
                                r, os = reward.halite_reward(ns, ship_action, ships[ship_id][1], next_obs['players'][player_id][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                                if not os:
                                    target[ship_action] = r + np.amax(model(next_state, training=False)[0]) * gamma
                            else:
                                r, os = reward.halite_reward(ns, ship_action, ships[ship_id][1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                                if not os:
                                    target[ship_action] = r

                        # inputs_data.append(state[0])
                        inputs_image.append(state)
                        targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        p = np.random.permutation(len(nn_inputs_image))
        nn_inputs_image = nn_inputs_image[p]
        nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size)

def train_on_games_sparse(model, folder, files, input_data_fnc, win_reward=10000, gamma=0.9, img_size=21, nbr_imgs=8, action_space=5, epochs=5, batch_size=32, sample_rate=0.1, size=21):
    inputs_data = []
    inputs_image = []
    targets = []
    player_id = 0

    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        game = game_data['game']
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0
        won = win_reward if winning_player == 0 else 0

        # print(len(game))

        for index, (obs, actions, other_ship_actions) in enumerate(game):
            ships = obs['players'][0][2]
            shipyards = obs['players'][0][1]
            # Looping over ships
            for ship_id, ship_info in ships.items():
                last_frame = False
                # Only prepare sample_rate of samples
                if sample_rate > np.random.rand():
                    # If not end of game and ship in next_ships
                    if index < len(game) - 1:
                        next_ships = game[index + 1][0]['players'][0][2].keys()
                        if ship_id in next_ships:
                            ns = True
                            next_obs = game[index + 1][0]
                            own_ship_actions = game[index + 1][2][ship_id]
                            next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                        else:
                            ns = False
                    else:
                        last_frame = True
                        ns = False

                    # Updating moved ships
                    what_does_our_ships = other_ship_actions[ship_id]
                    next_ship_poses = helper.get_next_ship_poses(ships, what_does_our_ships, ship_id)

                    if ship_id in actions:
                        ship_action = helper.ship_action_index(actions[ship_id])
                    else:
                        ship_action = 0

                    next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)
                    danger_poses = helper.get_danger_poses(obs, ship_info)
                    
                    # Getting state
                    state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)

                    # Target and reward calculations
                    target = np.array(model(state, training=False)[0])
                    if ship_action != 5:
                        if ns:
                            r, os = reward.halite_reward(ns, ship_action, ships[ship_id][1], next_obs['players'][0][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            if not os:
                                target[ship_action] = r + np.amax(model(next_state, training=False)[0]) * gamma
                        else:
                            r, os = reward.halite_reward(ns, ship_action, ships[ship_id][1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, target[ship_action])
                            if not os:
                                target[ship_action] = r

                        if last_frame:
                            potential_kills = helper.get_kill_poses(obs, ship_info, 1)

                            if len(game) == 399:
                                target[ship_action] += won
                            elif next_ship_pos in potential_kills:
                                target[ship_action] += won
                                print("maybe won game?")


                    # inputs_data.append(state[0])
                    inputs_image.append(state)
                    targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        p = np.random.permutation(len(nn_inputs_image))
        nn_inputs_image = nn_inputs_image[p]
        nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size)

def train_on_games_fully_sparse(model, folder, files, input_data_fnc, gamma=0.9, img_size=21, nbr_imgs=8, action_space=5, epochs=5, batch_size=32, sample_rate=0.1, size=21):
    inputs_data = []
    inputs_image = []
    targets = []
    player_id = 0

    for file in files: 
        with open(folder + '/' + file, 'rb') as pickle_file:
            game_data = pickle.load(pickle_file)

        game = game_data['game']
        result = game_data['players']

        player_status = helper.alive(result)
        winning_player = helper.player_won(result, player_status)
        losing_player = 1 if winning_player == 0 else 0
        won = 10000 if winning_player == 0 else 0

        # print(len(game))

        for index, (obs, actions, other_ship_actions) in enumerate(game):
            ships = obs['players'][0][2]
            shipyards = obs['players'][0][1]
            # Looping over ships
            for ship_id, ship_info in ships.items():
                last_frame = False
                # Only prepare sample_rate of samples
                if sample_rate > np.random.rand():
                    # If not end of game and ship in next_ships
                    if index < len(game) - 1:
                        next_ships = game[index + 1][0]['players'][0][2].keys()
                        if ship_id in next_ships:
                            ns = True
                            next_obs = game[index + 1][0]
                            own_ship_actions = game[index + 1][2][ship_id]
                            next_state, _ = input_data_fnc(next_obs, ship_id, player_id, own_ship_actions, size)
                        else:
                            ns = False
                    else:
                        last_frame = True
                        ns = False

                    # Updating moved ships
                    what_does_our_ships = other_ship_actions[ship_id]

                    if ship_id in actions:
                        ship_action = helper.ship_action_index(actions[ship_id])
                    else:
                        ship_action = 0

                    next_ship_pos = helper.get_next_ship_pos(ship_info[0], ship_action)

                    # Getting state
                    state, _ = input_data_fnc(obs, ship_id, player_id, what_does_our_ships, size)

                    # Target and reward calculations
                    target = np.array(model(state, training=False)[0])
                    if ship_action != 5:
                        if ns:
                            target[ship_action] = np.amax(model(next_state, training=False)[0]) * gamma

                        if last_frame:
                            potential_kills = helper.get_kill_poses(obs, ship_info, 1)

                            if len(game) == 399:
                                target[ship_action] += won
                            elif next_ship_pos in potential_kills:
                                target[ship_action] += won
                                print("maybe won game?")


                    # inputs_data.append(state[0])
                    inputs_image.append(state)
                    targets.append(target)

    if len(inputs_image) > 0:
        # nn_inputs_data = np.reshape(inputs_data, [len(inputs_data), self.nbr_data])
        nn_inputs_image = np.reshape(inputs_image, [len(inputs_image), img_size, img_size, nbr_imgs])
        nn_targets = np.reshape(targets, [len(targets), action_space])

        p = np.random.permutation(len(nn_inputs_image))
        nn_inputs_image = nn_inputs_image[p]
        nn_targets = nn_targets[p]
        
        model.fit(nn_inputs_image, nn_targets, epochs=epochs, batch_size=batch_size)