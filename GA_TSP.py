"""
author: Jianwei Zheng
time: 2020/05/10
TSP-遗传算法
核心思想：
（1）种群初始编码
（2）适应度定义
（3）选择操作：优秀的染色体被选中的概率和适应度函数成正比
（4）交叉操作：一对染色体的基因片段杂交
（5）变异操作：单个染色体的基因片段变异
（6）进化逆转操作：单个染色体的基因片段
（7）进化
"""

from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import sys


def hyper_parameter():
    max_evolution_num = 500     # 最大的进化树
    population_num = 500    # 种群数目
    cross_pro = 0.6     # 交叉概率
    mutation_pro = 0.1      #变异概率
    return max_evolution_num, population_num, cross_pro, mutation_pro


def distance(vec1, vec2):
    print('distance: ', np.linalg.norm(np.array(vec1) - np.array(vec2)))
    return np.linalg.norm(np.array(vec1) - np.array(vec2))


# 种群初始化编码（生成基因的随机排列）
def init_chromos(start=-1, gene_len=-1, population_num=0):
    chromos = []    # 所有种群
    for i in range(population_num):
        gene = list(range(gene_len))    # 染色体基因编码
        np.random.shuffle(gene)
        # 调换染色体的头部基因为给定的基因
        if start != -1:
            for j, g in enumerate(gene):
                if g == start:
                    gene[0], gene[j] = gene[j], gene[0]
        chromos.append(gene)
    return chromos


# 适应度函数，返回适应度分数
def calc_fin_ness(cities, gens):
    gens = np.copy(gens)
    gens = np.append(gens, gens[0])     # 在染色体的末尾添上头部基因
    D = np.sum([distance(cities[gens[i]], cities[gens[i+1]]) for i in range(len(gens) - 1)])
    return 1.0 / D


# 轮盘赌 (精英染色体被选中的概率与适应度函数打分的结果成正比
def roulette_gambler(fit_pros, chromos):
    pick = np.random.random()
    for j in range(len(chromos)):
        pick -= fit_pros[j]
        if pick <= 0:
            return j
    return 0


# 染色体选择操作（通过适应度函数打分和轮盘赌，来选择精英染色体）
def choice(cities, chromos):
    n = len(chromos)
    fit_pros = []
    [fit_pros.append(calc_fin_ness(cities, chromos[i])) for i in range(n)]
    choice_gens = []
    for i in range(n):
        j = roulette_gambler(fit_pros, chromos)     # 采用轮盘赌选择一个更好的染色体
        choice_gens.append(chromos[j])  # 选择一个染色体
    for i in range(n):
        chromos[i] = choice_gens[i]     # 优胜劣汰，替换出更精英的染色体
    return chromos


# 染色体交叉操作（两个染色体互相杂交基因片段，用于产生新的染色体，影响全局寻优能力）
def cross(chromos, cross_pro):
    gens_len = len(chromos[0])
    move = 0    # 当前基因移动的位置
    while move < gens_len - 1:
        cur_pro = np.random.random()    # 决定是否进行交叉操作
        # 本次不进行交叉操作
        if cur_pro > cross_pro:
            move += 2
            continue
        parent1, parent2 = move, move + 1   # 准备杂交的两个染色体
        index1 = np.random.randint(1, gens_len - 2)
        index2 = np.random.randint(index1, gens_len - 2)
        if index1 == index2:
            continue
        print('index1: ', index1)
        print('index2:', index2)
        print('parent1:', parent1)
        print('parent2:', parent2)
        print(chromos)
        # print('*****: ', chromos[104][84:185])
        temp_gen1 = chromos[parent1][index1:index2 + 1]   # 交换的基因片段1
        temp_gen2 = chromos[parent2][index1:index2 + 1]   # 交换的基因片段2
        print('temp_gen1:', chromos[parent1])
        print('temp_gen2:', chromos[parent2])
        # 杂交插入染色体片段
        temp_parent1, temp_parent2 = np.copy(chromos[parent1]).tolist(), np.copy(chromos[parent2]).tolist()
        temp_parent1[index1:index2+1] = temp_gen2
        temp_parent2[index1:index2+1] = temp_gen1
        # 消去冲突
        pos = index1 + len(temp_gen1)   # 插入杂交基因片段的结束位置
        conflict1_ids, conflict2_ids = [], []
        [conflict1_ids.append(i) for i, v in enumerate(temp_parent1) if v in temp_parent1[index1:pos]
         and i not in list(range(index1, pos))]
        [conflict2_ids.append(i) for i, v in enumerate(temp_parent2) if v in temp_parent2[index1:pos]
         and i not in list(range(index1, pos))]
        for i, j in zip(conflict1_ids, conflict2_ids):
            temp_parent1[i], temp_parent2[j] = temp_parent2[j], temp_parent1[i]
        chromos[parent1] = temp_parent1
        chromos[parent2] = temp_parent2
        move += 2
    return chromos


# 变异操作（随机调换单个染色体的基因位置）
def mutation(chromos, mutation_pro):
    n = len(chromos)
    gens_len = chromos[0]
    for i in range(n):
        cur_pro = np.random.random()    # 决定是否进行变异操作
        # 本次不进行变异操作
        if cur_pro > mutation_pro:
            continue
        index1 = np.random.randint(1, gens_len - 2)
        index2 = np.random.randint(1, gens_len - 2)
        chromos[i][index1], chromos[i][index2] = chromos[i][index2], chromos[i][index1]
    return chromos


# 逆转操作，让单个染色体变得更加优秀
def reverse(cities, chromos):
    n = len(chromos)
    gens_len = len(chromos[0])
    for i in range(n):
        flag = 0
        while flag == 0:
            index1 = np.random.randint(1, gens_len - 2)
            index2 = np.random.randint(index1, gens_len - 2)
            if index1 == index2:
                continue
            temp_chromos = np.copy(chromos[i])
            temp_chromos = temp_chromos.tolist()
            temp_gen = temp_chromos[index1:index2+1]
            temp_gen.reverse()
            temp_chromos[index1:index2+1] = temp_gen
            fit_score1 = calc_fin_ness(cities, chromos[i])
            fit_score2 = calc_fin_ness(cities, temp_chromos)
            # 说明经过逆转之后染色体将变得更加优秀
            if fit_score2 > fit_score1:
                chromos[i] = temp_chromos   # 更新染色体为逆染色体
            flag = 1
    return chromos


def GA(data):
    sys.stdout = open("GA_result.txt", "w")
    max_evolution_num, population_num, cross_pro, mutation_pro = hyper_parameter()
    cities = data
    print('type of cities: ', type(cities))
    print('type of cities[0]: ', type(cities[0]))
    best_gens = [-1 for _ in range(len(cities))]
    min_distance = np.inf   # 最短路径长度
    best_fit_index = 0  # 最短路径出现的代数
    start = 0   # 种群的初始位置

    # 开始进化
    for step in range(max_evolution_num):
        distance_arr = []   # 每一个染色体的总路程数组
        chromos = init_chromos(start=start, gene_len=len(cities), population_num=population_num)   # 种群初始化,得到所有种群
        chromos = choice(cities, chromos)   # 选择操作，选择出每个种群种群的精英染色体
        chromos = cross(chromos, cross_pro)    # 交叉操作，两个染色体相互杂交产生新的染色体
        chromos = mutation(chromos, mutation_pro)     # 变异操作，单个染色体变异
        chromos = reverse(cities, chromos)  # 变异操作，单个染色体变得更加优秀
        [distance_arr.append(1.0 / calc_fin_ness(cities, chromos[i])) for i in range(len(chromos))]
        best_gens_idx = np.argmin(distance_arr)     # 找到最短的路径位置，对应于精英染色体的位置
        if distance_arr[best_gens_idx] < min_distance:
            min_distance = distance_arr[best_gens_idx]  # 更新最短路径
            best_gens = chromos[best_gens_idx]      # 更新精英染色体
            best_gens.append(start)
            best_fit_index += 1

        print('通过{}代的基因进化，精英染色体出现在第{}代，基因序列为：'.format(max_evolution_num, best_fit_index))
        [print(chr(97 + v), end=',' if i < len(best_gens) - 1 else '\n') for i, v in enumerate(best_gens)]
        print('精英染色体映射的最短路径为：{}'.format(min_distance))