import socket
import struct
import numpy as np


class Connect:
    def __init__(self):
        self.ip = "localhost"
        self.server_port = 11111
        self.client_port = 6969

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_sock.bind((self.ip, self.server_port))

        self.client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def server(self):
        data, addr = self.server_sock.recvfrom(1024)
        info = np.fromstring(data, np.float32)
        return info

    def client(self, controls):
        self.client_sock.sendto(bytearray(struct.pack("f", controls)), (self.ip, self.client_port))


if __name__ == "__main__":
    connection = Connect()
    while True:
        print(connection.server())
        connection.client(0)