import numpy as np
import matplotlib.pyplot as plt 
import pdb
import sys
import time


def hyper_parameter():
    alpha = 0.99
    t = (1, 100)
    markov_len = 1000
    return alpha, t, markov_len


def SA(data, dis):
    sys.stdout = open("SA_whole_result.txt", "w")
    f = open("SA_result.txt", "w")
    alpha = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    t = [(1, 100), (1, 200), (1, 300), (1, 400), (1, 500), (1, 600), (1, 700)]
    l = [10, 50, 100, 500, 1000, 2000, 5000]
    result = ""
    for i in alpha:
        for j in t:
            for k in l:
                para = i, j, k
                each_result = patch_SA(data, dis, para)
                result += str(each_result) + "\n"
    f.write(str(result))

def patch_SA(data, dis, para):
    start_time = time.time()
    data = np.array(data)
    num = data.shape[0]
    distance = dis
    solution_new = np.arange(num)
    np.random.shuffle(solution_new)
    # value_new = np.max(num)

    solution_current = solution_new.copy()
    value_current = 100000000

    solution_best = solution_new.copy()
    value_best = 100000000
    # np.max

    alpha, t2, markovlen = para

    t = t2[1]

    result = []
    # 记录迭代过程中的最优解
    while t > t2[0]:
        for i in np.arange(markovlen):
            if np.random.rand() > 0.9:
                while True:
                    loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                    loc2 = np.int(np.ceil(np.random.rand()*(num-1)))
                    if loc1 != loc2:
                        break
                solution_new[loc1],solution_new[loc2] = solution_new[loc2],solution_new[loc1]
            else:
                while True:
                    loc1 = np.int(np.ceil(np.random.rand()*(num-1)))
                    loc2 = np.int(np.ceil(np.random.rand()*(num-1)))
                    loc3 = np.int(np.ceil(np.random.rand()*(num-1)))

                    if(loc1 != loc2) & (loc2 != loc3)&(loc1 != loc3):
                        break

                if loc1 > loc2:
                    loc1, loc2 = loc2, loc1
                if loc2 > loc3:
                    loc2, loc3 = loc3, loc2
                if loc1 > loc2:
                    loc1, loc2 = loc2, loc1

                tmp_list = solution_new[loc1:loc2].copy()
                solution_new[loc1:loc3-loc2+1+loc1] = solution_new[loc2:loc3+1].copy()
                solution_new[loc3-loc2+1+loc1:loc3+1] = tmp_list.copy()

            value_new = 0
            for i in range(num-1):
                value_new += distance[solution_new[i]][solution_new[i+1]]
            value_new += distance[solution_new[0]][solution_new[num-1]]
            if value_new < value_current:
                value_current = value_new
                solution_current = solution_new.copy()

                if value_new < value_best:
                    value_best = value_new
                    solution_best = solution_new.copy()
            else:
                if np.random.rand() < np.exp(-(value_new-value_current)/t):
                    value_current = value_new
                    solution_current = solution_new.copy()
                else:
                    solution_new = solution_current.copy()
        t = alpha*t
        result.append(value_best)
    end_time = time.time()
    print('*'*20)
    print('alpha: ', alpha)
    print('t: ', t2)
    print('l: ', markovlen)
    print('value_best: ', value_best)
    print('solution_best: ', solution_best)
    print('the total time is: ', end_time - start_time)
    # plt.plot(np.array(result))
    # plt.ylabel("the best value")
    # plt.xlabel("times")
    # plt.show()
    print(min(result))
    return str(alpha) + "," + str(t2[1]) + "," + str(markovlen) + "," + str(value_best) + "," + str(end_time - start_time)