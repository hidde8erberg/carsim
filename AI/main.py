# import tensorflow as tf
import numpy as np

from connect import Connect

if __name__ == '__main__':
    conn = Connect()
    while True:
        a, b, c = conn.server()
        if b > 65:
            conn.client(1)
