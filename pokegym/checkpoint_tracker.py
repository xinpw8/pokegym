import time

def checkpoint_met(map_n, wall_time):
    checkpoints = {
    "Viridian Forest Entrance": (51, None),
    "Viridian Forest Exit": (47, None),
    "Pewter": (2, None),
    "Badge 1": (54, None),
    "Mt Moon Entrance": (60, None),
    "Mt Moon Exit": (59, None),
    "Cerulean": (13, None),
    "Badge 2": (13, None),
    "Bill": (88, None),
    "Vermilion": (17, None),
    "SS Anne Start": (95, None)
}
    if map_n in checkpoints:
        a_time = time.time()
        timestamp = a_time - wall_time
        checkpoints[map_n] = (checkpoints[map_n][0], timestamp)
    
    return checkpoints