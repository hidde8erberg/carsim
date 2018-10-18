# import tensorflow as tf
import numpy as np

from connect import Connect
import utils

if __name__ == '__main__':
    conn = Connect()
    while True:
        a, b, c = utils.sort_server(np.array(conn.server()))
        print(a)
        conn.client(.5)
        if a:
            conn.client(0)
