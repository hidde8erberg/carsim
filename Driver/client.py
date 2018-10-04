import socket
import numpy as np


def client():
    ip = 'localhost'
    port = 22222

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    while True:
        data = np.array([1, 1, 1])
        sock.sendto(bytes(data), (ip, port))
