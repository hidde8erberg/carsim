import socket
# import numpy as np
import sys


def client():
    ip = 'localhost'
    port = 22222

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    while True:
        data = [1, 1, 1]
        sock.sendto(bytes(data), (ip, port))


if __name__ == '__main__':
    client()
