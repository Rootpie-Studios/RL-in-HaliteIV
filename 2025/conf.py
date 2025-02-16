def get(param):
    return {
        "size": 21,
        "episodeSteps": 400,
        "spawnCost": 500,
        "maxShips": 25,
        "debug": True,
    }[param]
