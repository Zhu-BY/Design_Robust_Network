import copy
import random as rd
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle

# def draw_G(G,next_G,pos,episode,Rc): # 绘制网络图
#     nx.draw(next_G, pos, node_size=20, edge_color='red')
#     nx.draw(G, pos, node_size=20,node_color = 'black')
#     edge_num = len(next_G.edges())
#     plt.savefig('episode%s_edges%s_Rc %.4f.jpg'%(episode,edge_num,Rc),dpi=600)
#     plt.close()
def generate_network(n,type,seed1):
    if type == 'er' or type=='ER':  # er临界网络
        G = nx.erdos_renyi_graph(n, n * 2 / (n * (n - 1)), seed=seed1)
    if type == 'er4' or type=='ER4': # k=4
        G = nx.erdos_renyi_graph(n, n*4  / (n * (n - 1)), seed=seed1)
    if type == 'ba' or type=='BA':  # m=1时，等价于tree网络
        m=1
        G = nx.barabasi_albert_graph(n,m,seed=seed1)
    if type == 'ba4' or type=='BA4':  # m=1时，等价于tree网络
        m=2
        G = nx.barabasi_albert_graph(n,m,seed=seed1)
    if type=='SF2.5':
        with open("../../Data/synthetic_network/"+"SF_graph_list_2.5.pkl", "rb") as f:
            graph_list = pickle.load(f)
        G = graph_list[0][seed1]
    if type=='SF2.1':
        with open("../../Data/synthetic_network/"+"SF_graph_list_2.1.pkl", "rb") as f:
            graph_list = pickle.load(f)
        G = graph_list[0][seed1]
    # pos = nx.spring_layout(G, seed=0)
    return G,0

def generate_network_cost(n,type,seed1):
    if type == 'er' or type=='ER':  # er临界网络
        G = nx.erdos_renyi_graph(n, n * 2 / (n * (n - 1)), seed=seed1)
    if type == 'er4' or type=='ER4': # k=4
        G = nx.erdos_renyi_graph(n, n*4  / (n * (n - 1)), seed=seed1)
    if type == 'ba' or type=='BA':  # m=1时，等价于tree网络
        m=1
        G = nx.barabasi_albert_graph(n,m,seed=seed1)
    if type == 'ba4' or type=='BA4':  # m=1时，等价于tree网络
        m=2
        G = nx.barabasi_albert_graph(n,m,seed=seed1)
    if type=='SF2.5':
        with open("../../Data/synthetic_network/N100/"+"SF_graph_list_2.5.pkl", "rb") as f:
            graph_list = pickle.load(f)
        G = graph_list[0][seed1]
    if type=='SF2.1':
        with open("../../Data/synthetic_network/N100/"+"SF_graph_list_2.1.pkl", "rb") as f:
            graph_list = pickle.load(f)
        G = graph_list[0][seed1]
    # pos = nx.spring_layout(G, seed=0)
    np.random.seed(seed1)
    random_sequence = list(np.random.uniform(1, 5, 100))
    return G, 0, random_sequence


def color_get():
    colors = [
        (202 / 255, 58 / 255, 69 / 255),  # 红0
        (67 / 255, 147 / 255, 164 / 255),  # 深蓝1
                 (144 / 255, 118 / 255, 115 / 255),  # 棕色2        # (77 / 255, 197 / 255, 109 / 255),  # 绿2
        (253 / 255, 176 / 255, 147 / 255),  # 橘黄3
        (193 / 255, 222/255, 156/255),  # 浅绿4
        (121 / 255, 167 / 255, 199 / 255),  # 浅蓝5
        (214 / 255, 224 / 255, 237 / 255),  # 淡紫色6
    ]
    return colors

def load_csv_net(name):
    path = "../Data/real_network/raw_data/"
    if name in ["weights-dist",'ISP']:
        path = path +"weights-dist/"+'1239/'
        edges = pd.read_csv(path+"weights.intra",sep=' ',header = None)
        # nodes = set(list(edges[0])+list(edges[1]))
        edge_list = [(edges[0][i],edges[1][i]) for i in range(len(edges))]
        G=nx.Graph()
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        gcc = max(list(nx.connected_components(G)),key=len)
        subgraph = G.subgraph(gcc)
        G = nx.Graph(subgraph)
        G=nx.convert_node_labels_to_integers(G)
        pos = nx.spring_layout(G)
        # import torch
        # pos = torch.load(path+"%s_pos.pth"%name)
        return G, pos
    if name in ['Germany grid','Germany power']:
        path = path+"Germany grid/"
        nodes_pos = pd.read_csv(path+"nodes.csv",sep='\t')
        edges = pd.read_csv(path + "edges.csv",sep='\t')
        pos = dict()
        for i in range(len(nodes_pos)):
            pos[nodes_pos['Name'][i]] =np.array([nodes_pos['Lat'][i],nodes_pos['Lon'][i]])
        G= nx.Graph()
        G.add_nodes_from([name for name in list(nodes_pos['Name'])])###########增加这句话，然后后面不convert_to_interger
        edge_list = [(edges['From_node'][i],edges['To_node'][i]) for i in range(len(edges))]
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        node_mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        # 生成新的网络和位置
        H = nx.relabel_nodes(G, node_mapping)
        new_pos = {node_mapping[node]: position for node, position in pos.items()}
        return H,new_pos
    if name in ['Central Chilean power grid','Chilean power']:
        path = path+"Central Chilean power grid/"
        nodes_pos = pd.read_csv(path+"Reduced_node.csv")
        edges = pd.read_csv(path + "Reduced_edge.csv")
        pos = dict()
        for i in range(len(nodes_pos)):
            pos[nodes_pos['Id'][i]-1] =np.array([nodes_pos['Longitude'][i],nodes_pos['Latitude'][i]])
        G= nx.Graph()
        G.add_nodes_from([Id-1 for Id in list(nodes_pos['Id'])])###########增加这句话，然后后面不convert_to_interger
        edge_list = [(edges['Source'][i]-1,edges['Target'][i]-1) for i in range(len(edges))]
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G,pos
    if name in ['USAir97','US Air']:
        path = path+"USAir97/"
        import scipy.io
        M = scipy.io.mmread(path+"inf-USAir97.mtx")
        G = nx.from_scipy_sparse_array(M)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)
        pos = nx.spring_layout(G)
        return G,pos
    else:
        nodes_pos = pd.read_csv(path+name+"/nodes.csv")
        edges = pd.read_csv(path +name+"/edges.csv")
        pos = dict()
        for i in range(len(nodes_pos)):
            x = nodes_pos[' _pos'][i]
            x = x.strip('array([])')
            x = x.split(",")#根据‘，’来将字符串分割成单个元素
            x = list(map(float, x))#分离出来的单个元素也是字符串类型的，将其转成浮点
            x = np.array(x)
            # nodes_pos[' _pos'][i] = x
            pos[nodes_pos["# index"][i]] = x
        G= nx.Graph()
        G.add_nodes_from(list(nodes_pos['# index']))###########增加这句话，然后后面不convert_to_interger
        edge_list = [(edges['# source'][i],edges[' target'][i]) for i in range(len(edges))]
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G,pos
def calculate_metrics(G):  #计算介数、一阶度、二阶度和三阶度
    # 计算介数中心性
    betweenness = nx.betweenness_centrality(G)
    # 计算所有节点的一阶度、二阶度和三阶度
    degrees = dict(G.degree())
    second_order_degrees = {}
    third_order_degrees = {}

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        second_neighbors = set()
        third_neighbors = set()
        for neighbor in neighbors:
            second_neighbors.update(G.neighbors(neighbor))
        second_order_degrees[node] = len(second_neighbors)

        for neighbor in second_neighbors:
            third_neighbors.update(G.neighbors(neighbor))
        third_order_degrees[node] = len(third_neighbors)

    return betweenness, degrees, second_order_degrees, third_order_degrees

# 提取括号内第一个数字（年份）
import re
def time_before_2025(filename):
    match = re.search(r'\((\d+),', filename)
    if match:
        year = int(match.group(1))
        if year < 2025:
            return True
        else:
            return False
def draw_dismanting_curve(path,type,dismantling_name,G0, G_list, name_list,r_curve_list=0,x_max0=0):
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    plt.rcParams['font.family'] = 'Arial'  # 设置字体
    plt.rcParams['font.size'] = 16  # 设置字号
    tab20c = plt.cm.get_cmap('tab20c', 20)
    # colors1 = [tab20c(0), tab20c(4), tab20c(8), tab20c(12),tab20c(16)]
    colors3 = [tab20c(1), tab20c(5), tab20c(9), tab20c(13) ,tab20c(17)]
    colors2 = [tab20c(3), tab20c(7), tab20c(11), tab20c(15),tab20c(18)]
    colors = color_get()
    colors1 = [(144 / 255, 118 / 255, 115 / 255),(215 / 255, 136 / 255, 115 / 255),colors[3],colors[1]]  #
    # colors1 = [colors[1],colors[6],colors[5],colors[7]]  #
    colors1.reverse()
    colors2.reverse()
    colors3.reverse()
    N=len(G0)
    if r_curve_list==0:
        name_list = ['initial']+name_list
        N=len(G0)
        r_curve_list = []
        for G in [G0]+G_list:
            dismantling_number = 1 if dismantling_name not in ['GND','GNDR'] else 4
            r, curve,m = dismantle(dismantling_name,G, dismantling_number)  # 当前边收益
            r_curve_list.append([r,curve])

    # fig, ax = plt.subplots(figsize=(6, 6))
    fig, ax = plt.subplots(figsize = (7,12))
    for plot,color1,color2,color3 in zip(r_curve_list,colors1,colors2,colors3):
        x = np.linspace(0, len(plot[1]) - 1, len(plot[1]))
        x = [i/N for i in x]
        ax.plot(x,plot[1], color=color1, linewidth=1,alpha=0.8)
    markers = ['o','<','d','s']
    for plot,name,color1,color2,color3,marker in zip(r_curve_list,name_list,colors1,colors2,colors3,markers):
        x = np.linspace(0, len(plot[1]) - 1, len(plot[1]))
        x = [i/N for i in x]
        ax.scatter(x,plot[1], marker=marker, s=30, alpha=0.99, facecolors='none', edgecolors=color1,linewidth=1.1, label='%s: %.2f' % (name,plot[0]))

    legend = ax.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('none')  # 设置图例边框颜色
    # frame.set_edgecolor('none')  # 设置图例边缘颜色
    frame.set_alpha(0)  # 设置图例边框透明
    # legend = ax.legend(frameon=1)
    major_locator = MultipleLocator(base=0.5)  # 主刻度位置间隔
    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
    plt.gca().yaxis.set_major_locator(major_locator)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    major_locator = MultipleLocator(base=0.2)  # 主刻度位置间隔
    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
    plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
    plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
    plt.title('Network Disintegration Curve for %s'%dismantling_name)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Residual robustness') #Size of Largest Connected Component
    xmax = max([len(x[1]) for x in r_curve_list])/len(G0) if x_max0==0 else x_max0
    plt.xlim(-0.01,xmax+0.05)
    plt.ylim(0 ,1.05)
    # plt.legend()
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig(path+" %s %s.png"%(type,dismantling_name), dpi=600)
    plt.show()
    return [x[0] for x in r_curve_list] # 返回R值


import re
import json

def abstract_pos_from_pyvisg(input,output):
    # 1. 读取 HTML 文件
    with open(input, 'r', encoding='utf-8') as f:
        html = f.read()

    # 2. 用正则提取 nodes 的 JSON 部分
    match = re.search(r'nodes = new vis\.DataSet\((\[.*?\])\);', html, re.S)
    if match:
        nodes_json = match.group(1)
    else:
        raise ValueError("无法在HTML中找到nodes数据")

    # 3. 将 nodes 加载成列表
    nodes = json.loads(nodes_json)

    # 4. 提取 id, x, y
    pos_dict = {}
    for node in nodes:
        node_id = node['id']
        x = node.get('x')
        y = node.get('y')
        if x is not None and y is not None:
            pos_dict[node_id] = {'x': x, 'y': y}

    # 5. 保存成 JSON 文件
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(pos_dict, f, ensure_ascii=False, indent=2)

    print('已成功导出到 nodes_pos.json！')


if __name__=='__main__':
    """检查真实网络load_csv_net"""
    # for name in ["ISP","US Air",'Central Chilean power grid','Deltacom','euroroad','Germany grid','Kdl','new-york','savannah','US power','washington']:
    #     G,pos = load_csv_net(name)
    #     print(name,len(G),len(G.edges()))
    """生成合成网络数据集"""
    # for typ in  ['BA','ER']:
    #     for n in [100, 600, 1000, 5000]:
    #         for i in range(200):
    #             G,_ = generate_network(n,typ,i)
    #             with open("../../Data/synthetic_network/N%s/"%n+"Graph_%s_%s_%s.pkl"%(n,typ,i), "wb") as f:
    #                 pickle.dump(G, f)
    #
    # for typ in  ['BA4','ER4']:
    #     for n in [100]:
    #         for i in range(200):
    #             G,_ = generate_network(n,typ,i)
    #             with open("../../Data/synthetic_network/N%s/"%n+"Graph_%s_%s_%s.pkl"%(n,typ,i), "wb") as f:
    #                 pickle.dump(G, f)
    #
    # for typ in  ['BA','ER']:
    #     for n in [100]:
    #         for i in range(200):
    #             G,_,node_cost = generate_network_cost(n,typ,i)
    #             with open("../../Data/synthetic_network/N%s/"%n+"Graph_%s_%s_%s.pkl"%(n,typ,i), "wb") as f:
    #                 pickle.dump([G,node_cost], f)
    """从pyvis图中提取pos"""