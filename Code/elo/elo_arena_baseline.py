import tensorflow as tf
import random, pickle

import helper, conf

# Constants
mean_elo = 1500
elo_width = 400
k_factor = 64

# Setting up scoreboard
try:
    with open('baseline_elo_dict', 'rb') as pickle_file:
        baseline_elo_dict = pickle.load(pickle_file)
    print('Successfully loaded old scoreboard')
except:
    print('Failed loading scoreboard, using hardcoded backup')
    quit()


# Elo helpers
def expected_result(elo_a, elo_b):
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

def get_elo(winner_elo, loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def update_elo(baseline_elo_dict, w, l, w_new_elo, l_new_elo):
    baseline_elo_dict[w]['elo'] = w_new_elo
    baseline_elo_dict[w]['w'] += 1
    baseline_elo_dict[l]['l'] += 1
    
    if conf.get('focus_player') == False:
        baseline_elo_dict[l]['elo'] = l_new_elo

def get_least_played(baseline_elo_dict):
    return sorted(baseline_elo_dict.items(), key=lambda item: item[1]['w'] + item[1]['l'])[0]

with tf.device('/CPU:0'):
    # Main loop
    while True:
        # print(baseline_elo_dict)
        # keys = random.sample(list(baseline_elo_dict), 2)
        if conf.get('focus_player') != False:
            # p0 = conf.get('focus_player')
            p0 = min(baseline_elo_dict.items(), key=lambda x: x[1]['w'] + x[1]['l'])[0]
        else:
            p0 = random.choice(list(baseline_elo_dict))

        p1 = random.choice(list(baseline_elo_dict))
        while p0 == p1:
            p1 = random.choice(list(baseline_elo_dict))
        print(p0, 'vs', p1)

        winner_idx = helper.play_halite_elo(p0, p1)
        if winner_idx == 0:
            if p1 in baseline_elo_dict[p0]:
                baseline_elo_dict[p0][p1]['w'] += 1
            else:
                baseline_elo_dict[p0][p1] = {'w' : 1, 'l' : 0}
            if p0 in baseline_elo_dict[p1]:
                baseline_elo_dict[p1][p0]['l'] += 1
            else:
                baseline_elo_dict[p1][p0] = {'w' : 0, 'l' : 1}
            
            p0_new, p1_new = get_elo(baseline_elo_dict[p0]['elo'], baseline_elo_dict[p1]['elo'])
            update_elo(baseline_elo_dict, p0, p1, p0_new, p1_new)
        elif winner_idx == 1:
            if p0 in baseline_elo_dict[p1]:
                baseline_elo_dict[p1][p0]['w'] += 1
            else:
                baseline_elo_dict[p1][p0] = {'w' : 1, 'l' : 0}
            if p1 in baseline_elo_dict[p0]:
                baseline_elo_dict[p0][p1]['l'] += 1
            else:
                baseline_elo_dict[p0][p1] = {'w' : 0, 'l' : 1}
                
            p1_new, p0_new = get_elo(baseline_elo_dict[p1]['elo'], baseline_elo_dict[p0]['elo'])
            update_elo(baseline_elo_dict, p1, p0, p1_new, p0_new)
        else:
            pass
        
        sorted_baseline_elo_dict = sorted(baseline_elo_dict.items(), key=lambda item: item[1]['elo'], reverse=True)
        
        print('\nX-----------------------------------------------------------------------------X')
        print('{:<3}  {:<20}  {:^3}  {:>12}  {:>12}  {:>12}'.format('', 'SCOREBOARD', '', 'elo', 'wins', 'losses'))
        print()
        for i, e in enumerate(sorted_baseline_elo_dict):
            a = e[0]
            data_string = '{:<3}  {:<20}  {:^3}  {:>12}  {:>12}  {:>12}'.format(str(i+1) + '.', a.split('.')[0], '-', int(baseline_elo_dict[a]['elo']), baseline_elo_dict[a]['w'], baseline_elo_dict[a]['l'])
            print(data_string)
        print('X-----------------------------------------------------------------------------X')


        with open('baseline_elo_dict', 'wb') as pickle_file:
            pickle.dump(baseline_elo_dict, pickle_file)