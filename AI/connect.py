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

    def unsort_receive(self):
        data, addr = self.server_sock.recvfrom(1024)
        info = np.fromstring(data, np.float32)
        return info

    def send(self, controls):
        if controls > 1:
            controls = 1
        elif controls < -1:
            controls = -1
        self.client_sock.sendto(bytearray(struct.pack("f", controls)), (self.ip, self.client_port))

    def receive(self):
        unsorted = self.unsort_receive()
        length = len(unsorted)
        sensors = np.array(unsorted[:length - 2])
        sensors[sensors == 0] = 20
        #sensors = np.divide(sensors, 20)
        s_reversed = sensors[::-1]
        distance = unsorted[length - 2]
        if unsorted[length - 1] == 0:
            crash = False
        elif unsorted[length - 1] == 1:
            crash = True
        else:
            raise Exception(f"crash value not 0 or 1, got:{data[length - 1]}")
        return s_reversed, distance, crash
