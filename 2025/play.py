from kaggle_environments import make

from agent import agent

# from teacher import agent as teacher
from simple_agent import agent as simple_agent
import os

# Create the Halite environment with 4 players
env = make("halite", configuration={"size": 21, "episodeSteps": 400})

# Run a game with 4 instances of your agent
steps = env.run([agent, agent, agent, agent])

# Print final game state
print(f"Game completed in {len(steps)} steps")
for player_idx, player in enumerate(steps[-1][0].observation.players):
    print(f"Player {player_idx} final halite: {player[0]}")

# Specify the output path for the HTML file
output_path = os.path.join(os.getcwd(), "halite_game.html")

# Render the game and save the HTML file
html = env.render(mode="html", width=800, height=600)
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Game visualization saved to: {output_path}")
