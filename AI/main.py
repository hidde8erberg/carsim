# import tensorflow as tf
import numpy as np

from connect import Connect
import utils

if __name__ == '__main__':
    conn = Connect()
    while True:
        a, b, c = utils.sort_server(conn.server())
        print(a)
        print(b)
        print(c)
        if b > 65:
            conn.client(1)
