import matplotlib as plt
import numpy as np
from fix_data import MatrixSensorData

if __name__ == '__main__':
    all_data = MatrixSensorData()
    all_data.load_csv()
    pick_data0 = all_data.ary0_50dim[0]

    print(pick_data0)