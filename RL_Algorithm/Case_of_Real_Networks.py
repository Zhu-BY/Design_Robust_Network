import argparse
import torch
from pyvis.network import Network
import matplotlib
from Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model.PPO import PPO
import matplotlib.pyplot as plt
import copy
from itertools import count
import pickle
from utils.base_utils import draw_dismanting_curve,color_get
import networkx as nx
from RL_Algorithm.utils.rl_utils import edge_mask,virtual_node_g_old,create_edge_indices,action_to_edge
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_pyvis_graph(G0,G1,name,ratio,outpath,highlight_node=[],position=False):  #G0 删边后的网络
    if G0!=0:
        edge_adds = [edge for edge in G1.edges if edge not in G0.edges]
        edge_adds_nodes = [x[0] for x in edge_adds] + [x[1] for x in edge_adds]
    colors = color_get()
    # rgba_colors = [colors[1],colors[7]]  #
    # rgba_colors = [colors[0],colors[1]]  #
    rgba_colors = [(144 / 255, 118 / 255, 115 / 255),colors[1]]  #
    hex_colors = ['#%02x%02x%02x' % (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)) for  rgba_color in rgba_colors]
    other_color='gray' #hex_colors[0]
    initial_color = 'black'
    add_color = hex_colors[1] if 'Initial' in name else hex_colors[0]
    width1 = 8
    width2 = 2

    net = Network()
    # net.toggle_physics(True)  # 关闭物理引擎/自动布局
    net.toggle_physics(False) if position!=False else net.toggle_physics(True)
    # 添加点
    G1_nodes = list(G1.nodes())
    degrees = [G1.degree(n) for n in G1_nodes]
    sizes = [(d+1) * 3 for d in degrees]
    for node, size in zip(list(G1.nodes()), sizes):
        if highlight_node!=0:
            if int(node)==highlight_node:
                net.add_node(node,label='%s'%highlight_node, color='red', size=size)
            else:
                net.add_node(node, label='', color=initial_color, size=size)
        else:
            net.add_node(node,label='', color=initial_color, size=size)
    # 添加边
    for edge in G1.edges():
        if G0!=0:
            if edge in edge_adds:
                source, target = edge
                net.add_edge(int(source), int(target), width=width1, color=add_color)
            else:
                source, target = edge
                net.add_edge(int(source), int(target), width=width2, color=other_color)
        else:
            source, target = edge
            net.add_edge(int(source), int(target), width=width1, color=initial_color)
    # 设置节点位置
    if position!=False:
        import json
        with open("../Data/real_network/"+position, 'r') as f:
            positions = json.load(f)
        for node_id, pos in positions.items():
            if int(node_id) in list(G1.nodes()):
                net.get_node(int(node_id))['x'] = pos['x']
                net.get_node(int(node_id))['y'] = pos['y']
    net.show_buttons(filter_=['physics'])
    net.save_graph(outpath+'%s %s topology'%(name,ratio)+".html")
    print('finish')

def main(real_name):
    outpath = './Design_result/real_network_result/uniform_cost/'
    path = "../Data/real_network/"
    with open(path + "%s_data.pkl"%real_name, "rb") as f:
        G_dict = pickle.load(f)
    attack_types =['HDA']
    for attack in attack_types:
        ratios = [0.3,0.4,0.5]
        data_loaded = []
        for ratio in ratios:
            load_data = torch.load(outpath+'/%s/%s_%s_%s.pth'%(real_name,real_name,ratio,attack))
            best_design = [x for x in load_data if x[0]==max([x[0] for x in load_data])][0]
            data_loaded.append(best_design)
        G_original = G_dict[0]
        # # 绘制不同重连比例的网络瓦解曲线
        target_size = 1 if attack not in ['GND', 'GNDR'] else 4
        R0, curve0, _ = dismantle(attack, G_original, target_size, -1)
        if real_name=='ISP':net_name = 'Sprintlink(US)'
        else:net_name=real_name
        draw_dismanting_curve(outpath,net_name,attack,G_original,0,
                              ['origin']+ratios,[[R0,curve0]]+copy.deepcopy(data_loaded),0.5 if net_name=='Sprintlink(US)' else 0.3)

        # 绘制重连网络图：删边绿色，新边红色，其他灰色
        cmap = plt.get_cmap('Accent')
        rgba_colors = [cmap(0),cmap(1)]
        result = data_loaded[-1]
        G_original = G_original
        G0 = result[3]
        G1 = result[2]
        draw_pyvis_graph(G0, G_original, '%s Initial'%net_name, 0.5,outpath,position="%s_position.json"%real_name)
        draw_pyvis_graph(G0, G1, '%s Rewire'%net_name, 0.5,outpath,position="%s_position.json"%real_name)
        print('Finish')


if __name__=='__main__':
    for name in ["ISP",'Germany Grid']:
    # for name in ['Germany Grid']:
        main(name)