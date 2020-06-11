import numpy as np
import matplotlib.pyplot as plt
import pdb
import math
import SA_TSP as SA
import GA_TSP as GA
import NN_TSP as NN
import test as t
import time
data_path = "./data/a280.tsp"


def process_data(data_path):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            if(line[0].isdigit()):
                pre = line.split()[1]
                lat = line.split()[2]
                each = (int(pre), int(lat))
                data.append(each)
    return data


def get_distance(coordinates):
    num = np.array(coordinates).shape[0]  # num个坐标点
    distance = np.zeros((num, num))  # num X num距离矩阵
    for i in range(num):
        for j in range(i, num):
            distance[i][j] = distance[j][i] = np.linalg.norm(np.array(coordinates[i]) - np.array(coordinates[j]))
    return distance


def main():
    data = process_data(data_path)
    distance = get_distance(data)
    result = 0
    for i in range(distance.shape[0]-1):
        result += distance[i][i+1]
    result += distance[0][distance.shape[0]-1]
    # for i in distance:
    #     print(i)
    print(result)
    SA.SA(data, distance)
    # GA.GA(data)
    # t.test(data)


if __name__ == '__main__':
    main()