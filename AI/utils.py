import numpy as np


def sort_server(data):
    length = len(data)
    sensors = np.array(data[:length - 2])
    s_reversed = sensors[::-1]
    distance = data[length - 2]
    if data[length - 1] == 0:
        crash = False
    elif data[length - 1] == 1:
        crash = True
    else:
        raise Exception(f"crash value not 0 or 1, got:{data[length - 1]}")
    return s_reversed, distance, crash
