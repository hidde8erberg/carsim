import socket
import numpy as np


def client():
    ip = 'localhost'
    port = 22222

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    while True:
        sock.sendto(bytes(512), (ip, port))
