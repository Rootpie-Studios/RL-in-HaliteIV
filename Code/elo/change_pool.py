import pickle

# if input('Change 4 pool? (y/n)') == 'y':
#     dict_name = 'elo_dict_4'
# else:
#     dict_name = 'elo_dict'

dict_name = 'elo_dict'

with open(dict_name, 'rb') as pickle_file:
    elo_dict = pickle.load(pickle_file)


if input('Add or remove agent? (a/r)') == 'a':
    agent_name = input('Enter agent name:')
    elo_dict[agent_name] = {'elo':1500, 'w':0, 'l':0}
    print('Added', agent_name)
else:
    agent_name = input('Enter agent name:')
    elo_dict.pop(agent_name, None)
    print('Removed', agent_name)

print(elo_dict.keys())
with open(dict_name, 'wb') as pickle_file:
    pickle.dump(elo_dict, pickle_file)
