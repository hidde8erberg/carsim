import socket
import numpy as np


def server():
    ip = 'localhost'
    port = 11111

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    while True:
        data, addr = sock.recvfrom(1024)
        info = np.fromstring(data, np.float32)
        print(info)


if __name__ == '__main__':
    server()
