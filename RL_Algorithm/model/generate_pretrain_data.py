from itertools import count
import numpy as np
import pickle
from RL_Algorithm.utils.base_utils import generate_network
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
import copy
import random as rd
from multiprocessing import Pool
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def random_low_degree_strategy(c,n,type):
    print(n,c)
    G0, _= generate_network(n,type,c)
    G = copy.deepcopy(G0)
    # edges_max = int(4*n/2)  # 网络平均度<=4，最大边数小于等于2*节点数
    # if len(G0)==600:
    #     edges_max = 700  # 网络平均度<=4，最大边数小于等于2*节点数
    edges_max = len(G0.edges())+100
    edges_num_now = len(G.edges())
    node_list = list(G.nodes())
    G_list = [copy.deepcopy(G0)]
    # edges_addnum=100
    edges_addnum=edges_max-edges_num_now
    action_list = []
    for i in range(edges_addnum):
        degree_dict = dict(G.degree())
        degree_value = sorted(list(degree_dict.values()))
        low_degree_2 = degree_value[0:2] # 最低度的两个度值
        edge=None
        for time in range(100):
            u_list = [k for k in node_list if degree_dict[k]==low_degree_2[0]]
            u = rd.choice(u_list)
            v_list = [k for k in node_list if degree_dict[k]==low_degree_2[1] and k!=u]
            v = rd.choice(v_list)
            if (u,v) not in G.edges() and (v,u) not in G.edges():
                edge=(u,v)
                break
        if edge==None:
            for time in range(100):
                u_list = [k for k in node_list if degree_dict[k] == degree_value[0]]
                u = rd.choice(u_list)
                v_list = [k for k in node_list if degree_dict[k]==degree_value[2] and k!=u]
                v = rd.choice(v_list)
                if (u,v) not in G.edges() and (v,u) not in G.edges():
                    edge=(u,v)
                    break
        if edge==None:
            print(1)
        action_list.append(edge)
        G.add_edge(edge[0], edge[1])
        G_list.append(copy.deepcopy(G))
    return [c,G_list,action_list]

def low_degree_generate(count=300,gtype='BA',n=100,path = ''):  # sequence=1:生成一组不断增加边的图
    try:
        with open(path+'Pretrain_data_%s_%s_%s_low_degree_design.pickle' % (count,gtype,n), 'rb') as file:
            c_G_list= pickle.load(file)
    except:
        c_G_list = []
    c_set = list(range(len(c_G_list),count)) #
    for c in c_set:
        c=c+100
        c_G = random_low_degree_strategy(c,n,gtype)
        c_G_list.append(c_G)
        with open(path+'Pretrain_data_%s_%s_%s_low_degree_design.pickle' % (count,gtype,n), 'wb') as file:
            pickle.dump(c_G_list, file)
    return c_G_list

if __name__=="__main__":
    data_dict = low_degree_generate(300)

