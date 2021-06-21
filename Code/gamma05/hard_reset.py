import os, shutil
import gamma05.conf as conf

folders = [conf.get('GAMES_FOLDER'), conf.get('HTML_FOLDER'), conf.get('EXPLOIT_GAMES_FOLDER'), conf.get('EXPLOIT_HTML_FOLDER')]

for folder in folders:
    folder_content = os.listdir(folder)
    for f in folder_content:
        os.remove(folder + '/' + f)

shutil.rmtree(conf.get("SHIP_MODEL"))