
def sort_server(data):
    length = len(data)
    sensors = data[:length]
    distance = data[length]
    if data[length + 1] == 0:
        crash = False
    elif data[length + 1] == 1:
        crash = True
    else:
        raise Exception(f"crash value not 0 or 1, got:{data[length + 1]}")
    return sensors, distance, crash
