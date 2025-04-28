import random
import copy
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
# from joblib import Parallel, delayed
from multiprocessing import Pool
import numpy as np

def get_key_by_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None  # 如果没找到

def local_search(G0,G_list,attack,baseline=0):
    num_add_edges = len(G_list[0].edges())-len(G0.edges())
    # 可用边集合
    edge_G0 = list(G0.edges())
    edge_dict = dict()  # 边对应的action index  # 1:一步生成边序号  2:分两步依次生成两个节点   3:一步生成两个节点
    e = 0  # 可用总边数
    for i in range(0, len(G0) - 1):
        for j in range(i + 1, len(G0)):
            if (i,j) not in edge_G0:
                edge_dict[e] = [i, j]
                e += 1
    # 初始解集
    initial_solutions = []
    for G in G_list:
        init_solution = [0] * len(edge_dict)
        G_edges = list(G.edges())
        G0_edges = list(G0.edges())
        for edge in G_edges:
            if edge not in G0_edges:
                key = get_key_by_value(edge_dict, [min(edge[0],edge[1]),max(edge[0],edge[1])])
                init_solution[key]=1
        initial_solutions.append(init_solution)

    LS = Local_search(G0,edge_dict,num_add_edges,attack,initial_solutions,baseline)
    # 根据初始解集开展解搜索，参考遗传算法交叉变异
    R_G = LS.genetic_algorithm()
    return R_G

class Local_search():
    def __init__(self,
                 G0, edge_dict, num_add_edges, attack, initial_solutions,baseline=0,R_initial=1,
                 POP_SIZE=200,  # 种群数量 # 50
                 CROSS_RATE=1,  # 交叉的概率为1
                 MUTATION_RATE=0.5,  # 变异的概率
                 GENERATIONS=500):  # 代数):
        super(Local_search, self).__init__()

        self.G0 = G0
        self.edge_dict = edge_dict
        self.attack = attack
        self.dis_num = 1 if attack not in ['GND', 'GNDR'] else 4
        self.initial_solutions = initial_solutions
        self.R0 = R_initial
        self.baseline = baseline

        self.POP_SIZE = POP_SIZE  # 种群数量
        self.SEQ_LEN = len(edge_dict)  # 个体长度
        self.ONES_COUNT = num_add_edges  # 1 的数量
        self.CROSS_RATE = CROSS_RATE  # 交叉的概率为1
        self.MUTATION_RATE = MUTATION_RATE  # 变异的概率
        self.GENERATIONS = GENERATIONS  # 代数):
        # self.F = [] # 适应度列表
        self.population = None
        # =======监控种群最优适应度=======#
        self.stagnant_generations = 0
        self.best_fitness_prev = None
        # =======监控种群最优适应度=======#

    # 生成合法个体：长度100，20个1
    def generate_individual(self):
        # gene = [0] * self.SEQ_LEN
        # ones_indices = random.sample(range(self.SEQ_LEN), self.ONES_COUNT)
        # for idx in ones_indices:
        #     gene[idx] = 1
        # return gene
        gene = np.zeros(self.SEQ_LEN, dtype=int)
        ones_indices = np.random.choice(self.SEQ_LEN, self.ONES_COUNT, replace=False)
        gene[ones_indices] = 1
        return list(gene)

    # 初始化种群
    def init_population(self):
        population = []
        # 复制初始解
        r = (0.5 * self.POP_SIZE) / len(self.initial_solutions)
        population = int(r) * self.initial_solutions
        # 随机解
        while len(population) < self.POP_SIZE:
            population.append(self.generate_individual())
        return population

    # 适应度函数（示例）：让1集中在前面，比如惩罚后面出现的1
    def fitness(self, i, ind=[]):
        if ind == []:
            ind = self.population[i]
        ones_positions = [j for j, value in enumerate(ind) if value == 1]
        added_edges_list = [self.edge_dict[m] for m in ones_positions]
        G = copy.deepcopy(self.G0)
        G.add_edges_from(added_edges_list)
        R, __, __ = dismantle(self.attack, G, self.dis_num, txtpath=i)
        # self.F[i]=R
        return R

    # def fitness(self,ind):
    #     return sum(val * (self.SEQ_LEN - idx) for idx, val in enumerate(ind))

    # 局部搜索策略：对当前个体进行局部搜索，尝试找到更好的解
    def local_refine(self, ind):
        best_ind = ind[:]
        best_fit = self.fitness(-1, ind)
        for _ in range(10):
            mutated = self.mutate(best_ind)
            fit = self.fitness(-1, mutated)
            if fit > best_fit:
                best_ind = mutated
                best_fit = fit
        return best_ind, best_fit

    # 选择（轮盘赌）
    def select(self, population, fitnesses, k):
        elite_k = max(1, int(0.1 * k))  # 精英个体保留
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        elite = [ind for ind, fit in sorted_pop[:elite_k]]

        total_fit = sum(fitnesses)
        probs = [f / total_fit for f in fitnesses]
        selected_rest = random.choices(population, weights=probs, k=k - elite_k)

        return elite + selected_rest

    # # 直接选择最优
    # def select_best(self,population, fitnesses, k):
    #     sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    #     return [ind for ind, fit in sorted_pop[:k]]

    # 交叉：双亲交换部分片段，再修复成合法个体
    def crossover(self, p1, p2):
        if random.random() > self.CROSS_RATE:
            return p1[:], p2[:]
        point = random.randint(1, self.SEQ_LEN - 2)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return self.repair(c1), self.repair(c2)

    def crossover_multi_point(self, p1, p2):
        num_points = random.randint(2, 10)
        points = sorted(random.sample(range(1, self.SEQ_LEN - 1), num_points))
        c1, c2 = p1[:], p2[:]
        for i in range(len(points) - 1):
            if i % 2 == 0:
                c1[points[i]:points[i + 1]] = p2[points[i]:points[i + 1]]
                c2[points[i]:points[i + 1]] = p1[points[i]:points[i + 1]]
        return self.repair(c1), self.repair(c2)

    def mutate(self, ind):
        strategies = [self.mutate_single_swap,
                      self.mutate_multi_swap,
                      self.mutate_shift,
                      self.mutate_resample]
        strategy = random.choice(strategies)
        return strategy(ind)

    # 变异：随机换两个位置（0变1，1变0），保证1的数量不变
    def mutate_single_swap(self, ind):
        ind = ind[:]
        ones = [i for i, v in enumerate(ind) if v == 1]
        zeros = [i for i, v in enumerate(ind) if v == 0]
        if ones and zeros:
            i1 = random.choice(ones)
            i0 = random.choice(zeros)
            ind[i1], ind[i0] = 0, 1
        return ind

    # 变异：随机交换多个1-0对
    def mutate_multi_swap(self, ind, k=3):
        ones = [i for i, v in enumerate(ind) if v == 1]
        zeros = [i for i, v in enumerate(ind) if v == 0]
        ind = ind[:]
        for _ in range(min(k, len(ones), len(zeros))):
            i1 = random.choice(ones)
            i0 = random.choice(zeros)
            ind[i1], ind[i0] = 0, 1
        return ind

    # 变异：位置偏移变异
    def mutate_shift(self, ind):
        ind = ind[:]
        ones = [i for i, v in enumerate(ind) if v == 1]
        if not ones:
            return ind
        i = random.choice(ones)
        shift = random.choice([-2, -1, 1, 2])
        j = i + shift
        if 0 <= j < self.SEQ_LEN and ind[j] == 0:
            ind[i] = 0
            ind[j] = 1
        return ind

    # 变异：重采样
    def mutate_resample(self, ind, p=0.3):
        ind = ind[:]
        if random.random() > p:
            return ind
        return self.generate_individual()

    # 修复函数：修复不合法的染色体，确保恰好20个1
    def repair(self, gene):
        gene = gene[:]
        count_ones = sum(gene)
        if count_ones > self.ONES_COUNT:
            ones = [i for i, v in enumerate(gene) if v == 1]
            for i in random.sample(ones, count_ones - self.ONES_COUNT):
                gene[i] = 0
        elif count_ones < self.ONES_COUNT:
            zeros = [i for i, v in enumerate(gene) if v == 0]
            for i in random.sample(zeros, self.ONES_COUNT - count_ones):
                gene[i] = 1
        return gene

    def adaptive_mutation_rate(self, best_fitness, gen):
        if best_fitness == self.best_fitness_prev:
            self.stagnant_generations += 1
        else:
            self.stagnant_generations = 0
        if self.stagnant_generations >= 10:
            self.MUTATION_RATE = min(0.8, self.MUTATION_RATE + 0.3)
            gen = 0
        else:
            self.MUTATION_RATE = max(0.1, 1 - gen / self.GENERATIONS)
        self.best_fitness_prev = best_fitness
        return gen

    # 主流程
    def genetic_algorithm(self):
        self.population = self.init_population()  # 初始解
        # fitnesses = [self.fitness(ind) for ind in population]
        # 开启并行，来计算Fitness
        pool = Pool(32)
        i_list = list(range(len(self.population)))
        fitnesses = pool.map(self.fitness, i_list)
        # for gen in range(self.GENERATIONS):
        gen = 0
        gen_count = 0
        while True:
            self.MUTATION_RATE = max(0.1, 1 - gen / self.GENERATIONS)
            # 选择
            selected = self.select(self.population, fitnesses, self.POP_SIZE)
            # 生成下一代
            next_gen = []
            for i in range(0, self.POP_SIZE, 2):
                # c1, c2 = self.crossover(selected[i], selected[(i+1) % self.POP_SIZE])
                c1, c2 = self.crossover_multi_point(selected[i], selected[(i + 1) % self.POP_SIZE])
                next_gen.append(self.mutate(c1))
                next_gen.append(self.mutate(c2))
            self.population = selected + next_gen[:self.POP_SIZE]
            # fitnesses = [self.fitness(ind) for ind in population]
            # 并行计算适应度
            i_list = list(range(len(self.population)))
            fitnesses = pool.map(self.fitness, i_list)
            # fitnesses = self.F[:]
            best = self.population[fitnesses.index(max(fitnesses))]
            if gen % 5 == 0:
                refined, refined_fit = self.local_refine(best)
                self.population.append(refined)
                # fitnesses.append(self.fitness(refined))
                fitnesses.append(refined_fit)
                best = self.population[fitnesses.index(max(fitnesses))]
            gen = self.adaptive_mutation_rate(max(fitnesses), gen)
            # print(f"Gen {gen_count + 1} | Best fitness: {max(fitnesses) / self.R0}")
            gen += 1
            gen_count += 1
            if gen_count >= self.GENERATIONS:
                # if gen>=self.GENERATIONS:
                if max(fitnesses) / self.R0 > self.baseline:
                    break
            if gen % 50 == 0:
                for _ in range(int(0.2 * self.POP_SIZE)):
                    self.population[random.randint(0, self.POP_SIZE - 1)] = self.generate_individual()

        ones_positions = [j for j, value in enumerate(best) if value == 1]
        added_edges_list = [self.edge_dict[i] for i in ones_positions]
        G = copy.deepcopy(self.G0)
        G.add_edges_from(added_edges_list)
        R, curve, __ = dismantle(self.attack, G, self.dis_num, txtpath=i)
        R_G = [R, curve, copy.deepcopy(G), copy.deepcopy(self.G0)]
        return R_G


def local_search_cost(G0,G_list,attack,sum_cost,node_cost,baseline=0):
    # num_add_edges = len(G_list[0].edges())-len(G0.edges())
    # 可用边集合
    edge_G0 = list(G0.edges())
    edge_dict = dict()  # 边对应的action index  # 1:一步生成边序号  2:分两步依次生成两个节点   3:一步生成两个节点
    e = 0  # 可用总边数
    for i in range(0, len(G0) - 1):
        for j in range(i + 1, len(G0)):
            if (i,j) not in edge_G0:
                edge_dict[e] = [i, j]
                e += 1
    # 初始解集
    initial_solutions = []
    for G in G_list:
        # init_solution = [0] * len(edge_dict)
        init_solution = []
        G_edges = list(G.edges())
        G0_edges = list(G0.edges())
        for edge in G_edges:
            if edge not in G0_edges:
                key = get_key_by_value(edge_dict, [min(edge[0],edge[1]),max(edge[0],edge[1])])
                init_solution.append(key)
        # residual_edges = [x for x in range(len(edge_dict)) if x not in init_solution]
        # rd.shuffle(residual_edges)
        initial_solutions.append(init_solution)

    LS = Local_search_cost(G0,edge_dict,sum_cost,attack,initial_solutions,baseline,node_cost)
    # 根据初始解集开展解搜索，参考遗传算法交叉变异
    R_G = LS.genetic_algorithm()
    return R_G


class Local_search_cost():
    def __init__(self,
        G0,edge_dict,sum_cost,attack,initial_solutions,baseline=0,node_cost=[],R_initial=1,
        POP_SIZE = 200,         # 种群数量
        CROSS_RATE = 1,       # 交叉的概率为1
        MUTATION_RATE = 0.2, # 变异的概率
        GENERATIONS = 500):    # 代数):
        super(Local_search_cost, self).__init__()

        self.G0=G0
        self.edge_dict = edge_dict
        self.edge_index_list = list(edge_dict.keys())
        self.attack = attack
        self.dis_num = 1 if attack not in ['GND', 'GNDR'] else 4
        self.initial_solutions = initial_solutions
        self.R0 = R_initial
        self.baseline=baseline
        self.node_cost = node_cost
        self.sum_cost = sum_cost

        self.POP_SIZE =POP_SIZE         # 种群数量
        # self.SEQ_LEN = len(edge_dict)              # 个体长度
        # self.ONES_COUNT = num_add_edges     # 1 的数量
        self.CROSS_RATE = CROSS_RATE       # 交叉的概率为1
        self.MUTATION_RATE =MUTATION_RATE # 变异的概率
        self.GENERATIONS = GENERATIONS    # 代数):

    # 初始化种群
    def init_population(self):
        # return [self.generate_individual() for _ in range(self.POP_SIZE)]
        r = self.POP_SIZE/len(self.initial_solutions)
        init_population = int(r+1)*self.initial_solutions
        return init_population
    # 适应度函数（示例）：让1集中在前面，比如惩罚后面出现的1
    def fitness(self,ind):
        # ones_positions = [j for j, value in enumerate(ind) if value == 1]
        # added_edges_list = [self.edge_dict[i] for i in ones_positions]
        # G=copy.deepcopy(self.G0)
        # G.add_edges_from(added_edges_list)
        added_edges_list = [self.edge_dict[i] for i in ind]
        G=copy.deepcopy(self.G0)
        now_cost = self.sum_cost
        for edge in added_edges_list:
            now_cost -=(self.node_cost[edge[0]]+self.node_cost[edge[0]])
            if now_cost<0:
                break
            G.add_edge(edge[0],edge[1])
        R,__,__=dismantle(self.attack,G,self.dis_num,txtpath=-999)
        return R

    # def fitness(self,ind):
    #     return sum(val * (self.SEQ_LEN - idx) for idx, val in enumerate(ind))

    # 选择（轮盘赌）
    # def select(pop, fitnesses):
    #     total = sum(fitnesses)
    #     probs = [f / total for f in fitnesses]
    #     return random.choices(pop, weights=probs, k=POP_SIZE)

    # 直接选择最优
    def select_best(self,population, fitnesses, k):
        sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
        return [ind for ind, fit in sorted_pop[:k]]

    # 交叉：双亲交换部分片段，再修复成合法个体
    def crossover(self,p1, p2):
        if random.random() > self.CROSS_RATE:
            return p1[:], p2[:]
        point = random.randint(1, min(len(p1),len(p2)) - 2)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return self.repair(c1), self.repair(c2)

    # 变异：随机换两个位置（0变1，1变0），保证1的数量不变
    def mutate(self,ind):
        if random.random() > self.MUTATION_RATE:
            return ind
        # i0,i1 = random.sample(list(range(len(ind))),2)
        # ind[i1], ind[i0] =ind[i0], ind[i1]
        old_edge_position = random.choice(list(range(len(ind))))
        while True:
            new_edge = random.choice(self.edge_index_list)
            if new_edge not in ind:
                ind[old_edge_position]=new_edge
                break
        return ind
    # 修复函数：修复不合法的染色体，确保恰好20个1
    def repair(self,gene):
        gene = gene[:]
        gene = list(set(gene))
        now_cost = self.sum_cost
        for i in range(len(gene)):
            edge_ind = gene[i]
            edge = self.edge_dict[edge_ind]
            now_cost -=(self.node_cost[edge[0]]+self.node_cost[edge[0]])
            if now_cost<0:
                gene = gene[0:i]
                return gene
        if now_cost==0:
            return gene
        if now_cost>0:
            edge_list_shuffle = self.edge_index_list
            random.shuffle(edge_list_shuffle)
            for edge_ind in edge_list_shuffle:
                if edge_ind not in gene:
                    edge = self.edge_dict[edge_ind]
                    now_cost -= (self.node_cost[edge[0]] + self.node_cost[edge[0]])
                    if now_cost<=0:
                        return gene
                    else:
                        gene.append(edge_ind)

    # 主流程
    def genetic_algorithm(self):
        population = self.init_population() # 初始解
        fitnesses = [self.fitness(ind) for ind in population]
        # for gen in range(self.GENERATIONS):
        gen=0
        while True:
            # 选择
            selected = self.select_best(population, fitnesses,self.POP_SIZE)
            # 生成下一代
            next_gen = []
            for i in range(0, self.POP_SIZE, 2):
                c1, c2 = self.crossover(selected[i], selected[(i+1) % self.POP_SIZE])
                next_gen.append(self.mutate(c1))
                next_gen.append(self.mutate(c2))
            population = selected + next_gen[:self.POP_SIZE]
            fitnesses = [self.fitness(ind) for ind in population]
            best = population[fitnesses.index(max(fitnesses))]
            print(f"Gen {gen + 1} | Best fitness: {max(fitnesses)/self.R0}")
            gen+=1
            if gen>=self.GENERATIONS:
                if max(fitnesses)/self.R0>=self.baseline:
                    break

        added_edges_list = [self.edge_dict[i] for i in best]
        G=copy.deepcopy(self.G0)
        now_cost = self.sum_cost
        for edge in added_edges_list:
            now_cost -=(self.node_cost[edge[0]]+self.node_cost[edge[0]])
            if now_cost<0:
                break
            G.add_edge(edge[0],edge[1])
        # R,__,__=dismantle(self.attack,G,self.dis_num,txtpath=i)
        # return [R,G]
        R, curve, __ = dismantle(self.attack, G, self.dis_num, txtpath=-1)
        R_G = [R, curve, copy.deepcopy(G), copy.deepcopy(self.G0)]
        return R_G


if __name__ == "__main__":
    LS = Local_search()
    LS.genetic_algorithm()