# import tensorflow as tf
import numpy as np

from connect import Connect
import utils

if __name__ == '__main__':
    conn = Connect()
    while True:
        print(conn.server())
        
        if conn.server()[5] > 65:
            conn.client(1)
