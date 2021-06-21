import matplotlib.pyplot as plt
import pickle, os
import numpy as np
import tensorflow as tf

import src.reward as reward
import src.helper as helper
import src.conf as conf

def sliding_avg_back(a, window_size):
    ret = []
    for i in range(len(a)):
        if i < window_size:
            avg = sum(a[:i+1]) / (i + 1)
            ret.append(avg)
        else:
            try:
                avg = sum(a[i-window_size:i])/window_size
                ret.append(avg)
            except:
                pass

    return ret
    
def sliding_avg_forward(a, window_size):
    ret = []
    for i in range(len(a)):
        if i+window_size < len(a):
            avg = sum(a[i:i+window_size])/window_size
            ret.append(avg)

    return ret

def plot_progress(window_size, folder, name, enemy, model, input_data_fnc, size=21, title=False, reward_fnc=reward.halite_reward):
    with tf.device('/CPU:0'):
        files = os.listdir(folder)
        files.sort(key=helper.natural_keys)

        nbr_files_to_plot = len(files)
        if input('Plot all? (y/n)\n') == 'n':
            nbr_files_to_plot = int(input('How many files to plot?\n'))
        files = files[:min(len(files), nbr_files_to_plot)]
        
        enemy = enemy.split('/')[1]
        print(enemy)

        final_halites = []
        final_cargos = []
        killed_ships = []
        killed_friends = []
        wins = []
        game_lens = []
        expected_rewards = []
        current_rewards = []
        
        for i, file in enumerate(files):
            try:
                print('Plotting game {}'.format(i))
                with open(folder + '/' + file, 'rb') as pickle_file:
                    game_data = pickle.load(pickle_file)

                game = game_data['game']
                result = game_data['players']

                player_status = helper.alive(result)
                winning_player = helper.player_won(result, player_status)
                p0_won = 1 if winning_player==0 else 0
                wins.append(p0_won)

                enemy_kills = 0
                friend_kills = 0
                frame_current_reward = 0

                for index, (obs, actions, other_ship_actions) in enumerate(game):
                    ships = obs['players'][0][2]
                    shipyards = obs['players'][0][1]
                    current_reward = 0
                    
                    for (ship_id, ship_info) in ships.items():

                        if ship_id in actions:
                            action = helper.ship_action_index(actions[ship_id])
                        else:
                            action = 0

                        current_ship_view = other_ship_actions[ship_id]
                        next_ship_poses = helper.get_next_ship_poses(ships, current_ship_view, ship_id)

                        next_ship_pos = helper.get_next_ship_pos(ship_info[0], action)

                        if index < len(game) - 1:
                            last_frame = False
                            next_obs = game[index + 1][0]
                            next_ships = next_obs['players'][0][2]
                            if ship_id in next_ships:
                                ns = True
                                # enemy_kills += reward.destroyed_enemy_cargo_reward(obs, next_obs, ship_id) // 500
                                if reward.destroyed_enemy_ship(obs, next_obs, ship_id):
                                    enemy_kills += 1

                            else:
                                ns = False

                            if next_ship_pos in next_ship_poses:
                                friend_kills += 1

                        else:
                            last_frame = True
                            ns = False

                        danger_poses = helper.get_danger_poses(obs, ship_info, enemy_id=1)

                        if action != 5:
                            if ns == True:
                                current_reward += reward_fnc(ns, action, ship_info[1], next_obs['players'][0][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, None)[0]
                            else:
                                current_reward += reward_fnc(ns, action, ship_info[1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, 0)[0]
                    
                    if len(ships) > 0:
                        frame_current_reward += current_reward/len(ships)

                current_rewards.append(frame_current_reward)
                
                final_halite = game[-1][0]['players'][0][0]

                final_halites.append(final_halite)
                final_cargos.append(helper.get_player_tot_cargo(obs['players'], 0))
                killed_ships.append(enemy_kills)
                killed_friends.append(friend_kills)
                game_lens.append(len(game))
                try:
                    expected_rewards.append(game_data['reward'])
                except:
                    pass

            except:
                pass


    # Halite and cargo
    plt.figure()
    avg_final_halite = sliding_avg_back(final_halites, window_size)
    avg_cargo_halites = sliding_avg_back(final_cargos, window_size)
    plt.plot(avg_final_halite, label='Avg final halite', color='C0')
    plt.plot(final_halites, label='Final halite', color='C0', alpha=0.2)
    plt.plot(avg_cargo_halites, label='Avg final cargo', color='C1')
    plt.plot(final_cargos, label='Final cargo', color='C1', alpha=0.2)

    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))    
    plt.legend(loc='upper left')

    # Enemies and friends killed
    plt.figure()
    avg_killed = sliding_avg_back(killed_ships, window_size)
    avg_friends = sliding_avg_back(killed_friends, window_size)
    plt.plot(avg_killed, label='Avg enemies killed', color='C0')
    plt.plot(killed_ships, label='Enemies killed', color='C0', alpha=0.2)
    plt.plot(avg_friends, label = 'Avg friends killed', color='C1')
    plt.plot(killed_friends, label='Friends killed', color='C1', alpha=0.2)

    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Winrate
    plt.figure()
    avg_win = sliding_avg_back(wins, window_size)
    plt.plot(avg_win, label='Avg win rate past {} games'.format(window_size))
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Game length
    plt.figure()
    avg_game_len = sliding_avg_back(game_lens, window_size)
    plt.plot(avg_game_len, label='Avg game length')
    plt.plot(game_lens, label='Game lengths', color='C0', alpha=0.2)
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Reward
    plt.figure()
    avg_er = sliding_avg_back(expected_rewards, window_size)
    plt.plot(avg_er, label='Avg expected reward')
    plt.plot(expected_rewards, label='Expected reward', color='C0', alpha=0.2)
    plt.legend(loc='upper left')
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Reward
    plt.figure()
    avg_cr = sliding_avg_back(current_rewards, window_size)
    plt.plot(avg_cr, label='Avg actual reward')
    plt.plot(current_rewards, label='Actual reward', color='C0', alpha=0.2)
    plt.legend(loc='upper left')
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')



    plt.show()

def plot_progress_enemy_data(window_size, folder, name, enemy, model, input_data_fnc, size=21, title=False, reward_fnc=reward.halite_reward):
    with tf.device('/CPU:0'):
        files = os.listdir(folder)
        files.sort(key=helper.natural_keys)

        nbr_files_to_plot = len(files)
        if input('Plot all? (y/n)\n') == 'n':
            nbr_files_to_plot = int(input('How many files to plot?\n'))
        files = files[:min(len(files), nbr_files_to_plot)]
        
        enemy = enemy.split('/')[1]
        print(enemy)

        final_halites = []
        final_cargos = []
        killed_ships = []
        killed_friends = []
        wins = []
        game_lens = []
        expected_rewards = []
        current_rewards = []
        
        for i, file in enumerate(files):
            print('Plotting game {}'.format(i))
            with open(folder + '/' + file, 'rb') as pickle_file:
                game_data = pickle.load(pickle_file)

            game = game_data['game']
            result = game_data['players']

            player_status = helper.alive(result)
            winning_player = helper.player_won(result, player_status)
            p0_won = 1 if winning_player==0 else 0
            wins.append(p0_won)

            enemy_kills = 0
            friend_kills = 0
            frame_current_reward = 0

            for index, (obs, actions, other_ship_actions, p_id) in enumerate(game):
                ships = obs['players'][0][2]
                shipyards = obs['players'][0][2]
                current_reward = 0
                
                for (ship_id, ship_info) in ships.items():

                    if ship_id in actions:
                        action = helper.ship_action_index(actions[ship_id])
                    else:
                        action = 0

                    current_ship_view = other_ship_actions[ship_id]
                    next_ship_poses = helper.get_next_ship_poses(ships, current_ship_view, ship_id)

                    next_ship_pos = helper.get_next_ship_pos(ship_info[0], action)

                    if index < len(game) - 1:
                        last_frame = False
                        next_obs = game[index + 1][0]
                        next_ships = next_obs['players'][0][2]
                        if ship_id in next_ships:
                            ns = True
                            # enemy_kills += reward.destroyed_enemy_cargo_reward(obs, next_obs, ship_id) // 500
                            if reward.destroyed_enemy_ship(obs, next_obs, ship_id):
                                enemy_kills += 1
                        else:
                            ns = False
                        if next_ship_pos in next_ship_poses:
                            friend_kills += 1
                    else:
                        last_frame = True
                        ns = False

                    danger_poses = helper.get_danger_poses(obs, ship_info, enemy_id=1)

                    if action != 5:
                        if ns == True:
                            current_reward += reward_fnc(ns, action, ship_info[1], next_obs['players'][0][2][ship_id][1], last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, None)[0]
                        else:
                            current_reward += reward_fnc(ns, action, ship_info[1], None, last_frame, shipyards, next_ship_poses, next_ship_pos, danger_poses, obs['step'], file, 0)[0]
                
                if len(ships) > 0:
                    frame_current_reward += current_reward/len(ships)

            current_rewards.append(frame_current_reward)

            final_halite = game[-1][0]['players'][0][0]

            final_halites.append(final_halite)
            final_cargos.append(helper.get_player_tot_cargo(obs['players'], 0))
            killed_ships.append(enemy_kills)
            killed_friends.append(friend_kills)
            game_lens.append(len(game))
            try:
                expected_rewards.append(game_data['reward'])
            except:
                pass

    
    x = range(len(files))

    # Halite and cargo
    plt.figure()
    avg_final_halite = sliding_avg_back(final_halites, window_size)
    avg_cargo_halites = sliding_avg_back(final_cargos, window_size)
    plt.plot(x, avg_final_halite, label='Avg final halite', color='C0')
    plt.plot(x, final_halites, label='Final halite', color='C0', alpha=0.2)
    plt.plot(x, avg_cargo_halites, label='Avg final cargo', color='C1')
    plt.plot(x, final_cargos, label='Final cargo', color='C1', alpha=0.2)
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Enemies and friends killed
    plt.figure()
    avg_killed = sliding_avg_back(killed_ships, window_size)
    avg_friends = sliding_avg_back(killed_friends, window_size)
    plt.plot(x, avg_killed, label='Avg enemies killed', color='C0')
    plt.plot(x, killed_ships, label='Enemies killed', color='C0', alpha=0.2)
    plt.plot(x, avg_friends, label = 'Avg friends killed', color='C1')
    plt.plot(x, killed_friends, label='Friends killed', color='C1', alpha=0.2)

    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Winrate
    plt.figure()
    avg_win = sliding_avg_back(wins, window_size)
    plt.plot(x, avg_win, label='Avg win rate past {} games'.format(window_size))
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Game length
    plt.figure()
    avg_game_len = sliding_avg_back(game_lens, window_size)
    plt.plot(x, avg_game_len, label='Avg game length')
    plt.plot(x, game_lens, label='Game lengths', color='C0', alpha=0.2)
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Reward
    plt.figure()
    avg_er = sliding_avg_back(expected_rewards, window_size)
    plt.plot(avg_er, label='Avg expected reward')
    plt.plot(expected_rewards, label='Expected reward', color='C0', alpha=0.2)
    plt.legend(loc='upper left')
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')

    # Reward
    plt.figure()
    avg_cr = sliding_avg_back(current_rewards, window_size)
    plt.plot(avg_cr, label='Avg actual reward')
    plt.plot(current_rewards, label='Actual reward', color='C0', alpha=0.2)
    plt.legend(loc='upper left')
    if title:
        plt.title(title)
    else:
        plt.title(name + ' vs ' + enemy + ', epsilon=' + str(conf.get('AGENT2_EPSILON')))
    plt.legend(loc='upper left')


    plt.show()