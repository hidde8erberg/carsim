
def sort_server(data):
    sensors = data[:5]
    distance = data[5]
    if data[6] == 0:
        crash = False
    elif data[6] == 1:
        crash = True
    else:
        raise Exception(f"crash value not 0 or 1, got:{data[6]}")
    return sensors, distance, crash
