from threading import Thread
from server import server
from client import client

if __name__ == '__main__':
    server = Thread(target=server())
    client = Thread(target=client())
    client.start()
    server.start()
