import pickle

dict_name = 'elo_dict'

with open(dict_name, 'rb') as pickle_file:
    elo_dict = pickle.load(pickle_file)

for a, data in elo_dict.items():
    data['elo'] = 1500
    data['w'] = 0
    data['l'] = 0

with open(dict_name, 'wb') as pickle_file:
    pickle.dump(elo_dict, pickle_file)

print([k for k in elo_dict.keys()])
print('Reset', dict_name)