import sys
import pickle

elo_dict = {}

try:
    with open('elo_dict', 'rb') as pickle_file:
        elo_dict = pickle.load(pickle_file)
        print('Successfully loaded scoreboard')
except:
    print('Could not find scoreboard...')
    sys.exit()

total = 0
for key, value in elo_dict.items():
    total += (value['w'] + value['l'])

print(total)