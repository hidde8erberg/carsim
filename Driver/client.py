import socket
# import numpy as np


def client():
    ip = 'localhost'
    port = 6969

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.bind((ip, port))

    while True:
        sock.sendto(bytes("hello", "utf-8"), (ip, port))


if __name__ == '__main__':
    client()
