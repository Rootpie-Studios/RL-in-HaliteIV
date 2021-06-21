import time, os, random

import src.helper as helper
import src.model as model

def pipeline_fnc(ship_agent, ship_model, agent1, agent1_exploit, agent2, number_of_sessions, number_of_games_in_session, configuration, play_exploit, games, html, exploit_games, exploit_html, train_fnc):
    start_time = time.time()

    nbr_played = len(os.listdir(games))

    while nbr_played < number_of_sessions:
        for j in range(number_of_games_in_session):
            helper.play_halite(agent1, agent2, games, html, configuration)
            if play_exploit:
                helper.play_halite(agent1_exploit, agent2, exploit_games, exploit_html, configuration)
                
        train_fnc(ship_agent, games, ship_model)
        
        nbr_played = len(os.listdir(games))
        passed_time = time.time() - start_time
        print('Current model:', ship_model, '. Total time passed:', passed_time, '. Progress: ', (nbr_played/number_of_sessions)*100, '%')

def pipeline_fnc_multi_agent(ship_agent, ship_model, agent1, agent1_exploit, enemies, number_of_sessions, number_of_games_in_session, configuration, play_exploit, games, html, exploit_games, exploit_html, train_fnc):
    start_time = time.time()

    nbr_played = len(os.listdir(games))

    while nbr_played < number_of_sessions:
        agent2 = random.choice(enemies)
        print('Playing against ' + agent2)
        for j in range(number_of_games_in_session):
            helper.play_halite(agent1, agent2, games, html, configuration)
            if play_exploit:
                helper.play_halite(agent1_exploit, enemies[0], exploit_games, exploit_html, configuration)
                
        train_fnc(ship_agent, games, ship_model)
        
        nbr_played = len(os.listdir(games))
        passed_time = time.time() - start_time
        print('Current model:', ship_model, '. Total time passed:', passed_time, '. Progress: ', (nbr_played/number_of_sessions)*100, '%')

def pipeline_fnc_sparse(ship_agent, ship_model, agent1, agent1_exploit, enemies, number_of_sessions, number_of_games_in_session, configuration, play_exploit, games, html, exploit_games, exploit_html, train_fnc):
    start_time = time.time()

    nbr_played = len(os.listdir(games))

    while nbr_played < number_of_sessions:
        for j in range(number_of_games_in_session):
            agent2 = random.choice(enemies)
            helper.play_halite_sparse(agent1, agent2, games, html, configuration)
            if play_exploit:
                helper.play_halite_sparse(agent1_exploit, agent2, exploit_games, exploit_html, configuration)
                
        train_fnc(ship_agent, games, ship_model)
        
        nbr_played = len(os.listdir(games))
        passed_time = time.time() - start_time
        print('Current model:', ship_model, '. Total time passed:', passed_time, '. Progress: ', (nbr_played/number_of_sessions)*100, '%')