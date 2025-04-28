import copy
import networkx as nx
import numpy as np
from math import ceil,log
import igraph as ig
# import random as rd
import time
from collections import Counter
import random as rd
from itertools import count
from RL_Algorithm.Environment.dismantlers.interface_decycler import min_sum_with_seeds
from RL_Algorithm.Environment.dismantlers.interface_gnd import gnd_with_seeds

def get_neigbors(g, node, depth):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1,depth+1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x,[]))
        nodes = output[i]
    return output
def choose_CI2(G0,l=2):
    G=copy.deepcopy(G0)
    # igraph
    g = ig.Graph(directed=False)
    g.add_vertices(list(G.nodes))
    g.add_edges(list(G.edges))
    # 找到最大团
    CI = dict() # 计算当前网络的CI值
    Nl_1 = g.neighborhood(g.vs.indices,order=l-1)
    Nl = g.neighborhood(g.vs.indices,order=l)
    Ball_l = [[x for x in Nl[i] if x not in Nl_1[i]] for i in range(len(g.vs.indices))]
    for i in g.vs.indices:
        CI[i] = (g.degree()[i]-1)*sum([g.degree()[j]-1 for j in Ball_l[i]])
    max_CI = max(list(CI.values()))
    u_list = [k for k in list(CI.keys()) if CI[k] == max_CI]
    # node_with_CI_max_g = rd.choice(u_list)
    node_with_CI_max_g = u_list[0]
    node_with_CI_max_G = list(G.nodes())[node_with_CI_max_g]
    G.remove_node(node_with_CI_max_G)
    g.delete_vertices(node_with_CI_max_g)
    gcc = len(max(g.connected_components(), key=len))
    return G,node_with_CI_max_G,gcc

def choose_RB(G0):
    G=copy.deepcopy(G0)
    g = ig.Graph(directed=False)
    g.add_vertices(list(G.nodes))
    try:
        g.add_edges(list(G.edges))
    except:
        print(1)
        # igraph
    ig_betweenness = g.betweenness(directed=False)
    c= Counter(ig_betweenness)
    if c[max(ig_betweenness)]>1:
        node_with_bet_max_gs = [ind for ind in range(len(ig_betweenness)) if ig_betweenness[ind] ==max(ig_betweenness)]
        # rd.seed()
        # node_with_bet_max_g = rd.choice(node_with_bet_max_gs)
        node_with_bet_max_g = node_with_bet_max_gs[0]
    else:
        node_with_bet_max_g = ig_betweenness.index(max(ig_betweenness))
    node_with_bet_max_G = list(G.nodes())[node_with_bet_max_g]
    G.remove_node(node_with_bet_max_G)
    g.delete_vertices(node_with_bet_max_g)
    gcc = len(max(g.connected_components(), key=len)) # 当前最大连通子团
    return G,node_with_bet_max_G,gcc
def choose_remove_node(G0,attack):
    for node in list(G0.nodes):
        if G0.degree(node) == 0:
            G0.remove_node(node)
    G =nx.convert_node_labels_to_integers(G0)
    if attack in ['RD','HDA']:
        deg = dict(G.degree())
        max_deg = max(list(deg.values()))
        u_list = [k for k in list(deg.keys()) if deg[k] == max_deg]
        remove_node = u_list[0]
        G.remove_node(remove_node)
        gcc = len(list(max(nx.connected_components(G), key=len))) # 当前最大连通子团
    if attack in ['RB','HBA']:
        G,remove_node,gcc = choose_RB(G)
    if attack in ['CI2']:
        G, remove_node, gcc = choose_CI2(G)
    if attack == 'MS':
        _,_, remove_seeds = min_sum_with_seeds(G, 1, 1)
        remove_node = remove_seeds[0]
    if attack == 'GND':
        _,_, remove_seeds = gnd_with_seeds(G, 4, R=0)
        remove_node = remove_seeds[0]
    if attack == 'GNDR':
        _,_, remove_seeds = gnd_with_seeds(G, 4, R=1)
        remove_node = remove_seeds[0]
    if attack in ['MS','GNDR','GND']:
        G.remove_node(remove_node )
        gcc = len(list(max(nx.connected_components(G), key=len)))  # 当前最大连通子团
    return G,gcc

def mix_dis(G0,attacks = ['RD','RB','CI2','MS','GND','GNDR'],s=0):
    G = copy.deepcopy(G0)
    attacks = ['RD','RB','CI2','MS','GND','GNDR']
    rd.seed(s)
    attack_list1 = rd.choices(range(len(attacks)), k=len(G0))
    attack_list2 = rd.choices(range(4), k=len(G0)) # 当瓦解尺寸小于4时排除GND和GNDR
    gcc0 = len(list(max(nx.connected_components(G0), key=len)))
    gcc = gcc0
    r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
    for t in count():
        if gcc>20:
            attack = attacks[attack_list1[t]]
        else: # 如果gcc小于4，排除GND和GNDR
            attack = attacks[attack_list2[t]]
        G, gcc = choose_remove_node(G,attack)
        r_list.append(gcc/len(G0))
        if gcc==1:
            break
    # 计算 R 值
    Rc = sum(r_list)
    return Rc,r_list

if __name__=="__main__":
    print('hi')
    '''测试瓦解策略'''
    G = nx.erdos_renyi_graph(100,400/(100*99))
    a, b= mix_dis(G)
    print(1)

