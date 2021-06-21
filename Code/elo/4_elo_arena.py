import random, pickle

import helper, conf

# Constants
mean_elo = 1500
elo_width = 400
k_factor = 64

# Setting up scoreboard
try:
    with open('elo_dict_4', 'rb') as pickle_file:
        elo_dict_4 = pickle.load(pickle_file)
    print('Successfully loaded old scoreboard')
except:
    elo_dict_4 = {
        'baseline500.py'    :   {'elo':1500, 'w':0, 'l':0}, 
        'smaller_nn_551.py' :   {'elo':1500, 'w':0, 'l':0},
        'TOM.py'            :   {'elo':1500, 'w':0, 'l':0},
        'p26.py'            :   {'elo':1500, 'w':0, 'l':0}, 
        'p26_5.py'          :   {'elo':1500, 'w':0, 'l':0},
        'p26_10.py'         :   {'elo':1500, 'w':0, 'l':0}, 
        'p26_15.py'         :   {'elo':1500, 'w':0, 'l':0}, 
        'no_action.py'      :   {'elo':1500, 'w':0, 'l':0},
        'random'            :   {'elo':1500, 'w':0, 'l':0}
    }

    print('Failed loading scoreboard, using hardcoded backup')

# Elo helpers
def expected_result(elo_a, elo_b):
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

def get_elo_change(winner_elo, loser_elo):
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    return change_in_elo

def update_elo(elo_dict_4, a1, a2, a1_place, a2_place):
    if a1_place < a2_place:
        elo_change = get_elo_change(elo_dict_4[a1]['elo'], elo_dict_4[a2]['elo'])
        elo_dict_4[a1]['elo'] += elo_change
        elo_dict_4[a2]['elo'] -= elo_change
        elo_dict_4[a1]['w'] += 1
        elo_dict_4[a2]['l'] += 1
    else:
        elo_change = get_elo_change(elo_dict_4[a2]['elo'], elo_dict_4[a1]['elo'])
        elo_dict_4[a2]['elo'] += elo_change
        elo_dict_4[a1]['elo'] -= elo_change
        elo_dict_4[a2]['w'] += 1
        elo_dict_4[a1]['l'] += 1

def get_least_played(elo_dict_4):
    return sorted(elo_dict_4.items(), key=lambda item: item[1]['w'] + item[1]['l'])[0]


# Main loop
while True:
    p0, p1, p2, p3 = random.sample(list(elo_dict_4), 4)
    print(p0, 'vs', p1, 'vs', p2, 'vs', p3)

    placements = helper.play_4_halite_elo(p0, p1, p2, p3)

    # Updating elos
    for a1_place, a1 in enumerate(placements):
        for a2_place, a2 in enumerate(placements):
            if a1 != a2 and a1_place < a2_place:
                update_elo(elo_dict_4, a1[0], a2[0], a1_place, a2_place)
    
    sorted_elo_dict_4 = sorted(elo_dict_4.items(), key=lambda item: item[1]['elo'], reverse=True)
    
    print('\nX-----------------------------------------------------------------------------X')
    print('{:<3}  {:<15}  {:^3}  {:>12}  {:>12}  {:>12}'.format('', 'SCOREBOARD', '', 'elo', 'wins', 'losses'))
    print()
    for i, e in enumerate(sorted_elo_dict_4):
        a = e[0]
        data_string = '{:<3}  {:<15}  {:^3}  {:>12}  {:>12}  {:>12}'.format(str(i+1) + '.', a.split('.')[0], '-', int(elo_dict_4[a]['elo']), elo_dict_4[a]['w'], elo_dict_4[a]['l'])
        print(data_string)
    print('X----------------------------------------------------------------------------X')

    with open('elo_dict_4', 'wb') as pickle_file:
        pickle.dump(elo_dict_4, pickle_file)