import pickle

elo_dict = {
            # baseline
            'baseline_500_a.py':        {'elo':1500,    'w':0,      'l':0}, 
            'baseline_1000.py':         {'elo':1500,    'w':0,      'l':0}, 

            # baseline increments
            'baseline_50.py':           {'elo':1500,    'w':0,      'l':0}, 
            'baseline_100.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_150.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_200.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_250.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_300.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_350.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_400.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_450.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_500_b.py':        {'elo':1500,    'w':0,      'l':0}, 

            # hard coded
            'no_action.py':             {'elo':1500,    'w':0,      'l':0}, 
            'random':                   {'elo':1500,    'w':0,      'l':0},
            'p26.py':                   {'elo':1500,    'w':0,      'l':0}, 
            'p26_15.py':                {'elo':1500,    'w':0,      'l':0},
            'p4.py':                    {'elo':1500,    'w':0,      'l':0}, 
            'p6.py':                    {'elo':1500,    'w':0,      'l':0}, 

            # NN layout
            'smaller_nn.py':            {'elo':1500,    'w':0,      'l':0}, 
            'tiny_nn.py':               {'elo':1500,    'w':0,      'l':0}, 
            'tiny_deep_nn.py':          {'elo':1500,    'w':0,      'l':0}, 
            'wider_nn.py':              {'elo':1500,    'w':0,      'l':0}, 
            'deeper_nn.py':             {'elo':1500,    'w':0,      'l':0}, 

            # reward function change
            'to_sparse.py':             {'elo':1500,    'w':0,      'l':0}, 
            'with_sparse.py':           {'elo':1500,    'w':0,      'l':0}, 
            'with_sparse_pool1.py':     {'elo':1500,    'w':0,      'l':0}, 
            'fully_sparse.py':          {'elo':1500,    'w':0,      'l':0}, 
            'team_spirit.py':           {'elo':1500,    'w':0,      'l':0}, 
            'kill_penalty.py':          {'elo':1500,    'w':0,      'l':0}, 

            # enemy data
            'enemy_data.py':            {'elo':1500,    'w':0,      'l':0}, 

            # gamma
            'gamma01.py':               {'elo':1500,    'w':0,      'l':0}, 
            'gamma05.py':               {'elo':1500,    'w':0,      'l':0}, 
            'gamma099.py':              {'elo':1500,    'w':0,      'l':0}, 
            'gamma099_1000.py':         {'elo':1500,    'w':0,      'l':0}, 

            # epsilon
            'eps_zero.py':              {'elo':1500,    'w':0,      'l':0}, 
            'eps_decay.py':             {'elo':1500,    'w':0,      'l':0}, 

            #compacting input
            'compact6.py':              {'elo':1500,    'w':0,      'l':0}, 
            
            # nearsighted
            'nearsighted.py':           {'elo':1500,    'w':0,      'l':0}, 

            # other training opponents
            'baseline_p26.py':          {'elo':1500,    'w':0,      'l':0}, 
            'baseline_p4.py':           {'elo':1500,    'w':0,      'l':0},
            'baseline_self.py':         {'elo':1500,    'w':0,      'l':0}
        }

with open('elo_dict', 'wb') as pickle_file:
    pickle.dump(elo_dict, pickle_file)