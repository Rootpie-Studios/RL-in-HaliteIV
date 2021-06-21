from prettytable import PrettyTable
import sys
import pickle

table = PrettyTable()
elo_dict = {}

try:
    with open('elo_dict', 'rb') as pickle_file:
        elo_dict = pickle.load(pickle_file)
        print('Successfully loaded scoreboard')
except:
    print('Could not find scoreboard...')
    sys.exit()

agents = list(elo_dict.keys())

table.add_column('Agents, W/L', agents)

agent_w_l = {}

for agent, data in elo_dict.items():
    w_ls = []
    for opponent in agents:
        w_l = ''
        if opponent in data:
            w_l = str(data[opponent]['w']) + ' / ' + str(data[opponent]['l'])
        else:
            w_l = '0 / 0'

        w_ls.append(w_l)
    
    agent_w_l[agent] = w_ls

for agent in agents:
    table.add_column(agent, agent_w_l[agent])

print(table)
        

