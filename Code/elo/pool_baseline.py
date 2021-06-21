import pickle

baseline_elo_dict = {

            # baseline increments
            'baseline_50_inc.py':       {'elo':1500,    'w':0,      'l':0}, 
            'baseline_100_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_150_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_200_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_250_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_300_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_350_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_400_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_450_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_500_inc.py':      {'elo':1500,    'w':0,      'l':0}, 
            'baseline500.py'    :      {'elo':1500,    'w':0,      'l':0}, 
            'baseline_1000.py'   :      {'elo':1500,    'w':0,      'l':0}, 

        }

with open('baseline_elo_dict', 'wb') as pickle_file:
    pickle.dump(baseline_elo_dict, pickle_file)