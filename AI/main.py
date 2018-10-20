import numpy as np

from connect import Connect
from network import Network
import utils

if __name__ == '__main__':
    net = Network()
    conn = Connect()

    while True:
        a, b, c = utils.sort_server(np.array(conn.server()))
        print(b)
        conn.client(-.5)
        if c:
            print("------- crash --------")
