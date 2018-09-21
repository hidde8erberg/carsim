import socket
import numpy as np

ip = 'localhost'
port_receive = 11111
port_send = 22222

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((ip, port_receive))

while True:
    data, addr = sock.recvfrom(1024)
    distances = np.fromstring(data, np.float32)
    print(distances)

    sock.sendto(1, (ip, port_send))
    print("send successful")
