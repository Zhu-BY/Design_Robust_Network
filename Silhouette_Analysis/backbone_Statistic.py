import pandas as pd
from RL_Algorithm.Environment.dismantlers.dismantlers_with_seeds import dismantle_with_seeds
import torch
import pickle
import numpy as np
import random as rd
import copy
from RL_Algorithm.utils.base_utils import generate_network,calculate_metrics,color_get
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = 'Arial'  # 设置字体
plt.rcParams['font.size'] = 16  # 设置字号
colors = color_get()
colors.reverse()
custom_colormap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=10)
def draw_dismanting_curve(dismantling_name, G_list):
    r_curve_list = []
    remove_nodes_list = []
    for G in G_list:
        dismantling_number = 1 if dismantling_name not in ['GND','GNDR'] else 4
        r, curve,m,remove_nodes = dismantle_with_seeds(dismantling_name,G, dismantling_number)
        r_curve_list.append([r,curve])
        remove_nodes_list.append(remove_nodes)
    return [x[0] for x in r_curve_list],remove_nodes_list # 返回R
def statisics(graph_type='BA',attack='HDA',n=100):
    data_loaded = torch.load('../RL_Algorithm/Design_result/synthetic_result/N%s/%s_%s_%s_rl_result.pth' % (n,attack, graph_type, n))
    RL_list = [data_loaded[i][1] for i in range(100)]
    first_peak_data_list = []
    for id_ in range(0,100):
        G0, _ = generate_network(100, graph_type, id_)
        G_Rl = RL_list[id_]
        G_list = [G_Rl] # 6个
        """计算上述网络瓦解过程"""
        R_list_now, remove_nodes_list = draw_dismanting_curve(attack, G_list)
        """计算backbone"""
        G_step = copy.deepcopy(G_Rl)
        remove_nodes = remove_nodes_list[0]
        data = []
        # betweenness, degrees, second_order, third_order = calculate_metrics(G_step)
        # data.append(betweenness, degrees, second_order, third_order)
        step_num= 40 if attack !='MS' else 60
        """瓦解过程节点特征统计"""
        for step in range(step_num):
            try:
                G_step.remove_node(remove_nodes[step])
            except:
                break
            gcc = list(max(nx.connected_components(G_step),key=len))
            G_gcc = nx.Graph()
            G_gcc.add_nodes_from(gcc)
            for edge in G_step.edges():
                if edge[0] in gcc and edge[1] in gcc:
                    G_gcc.add_edge(edge[0],edge[1])
            # 瓦解过程的一阶度变化、二阶度、三阶度变化、介数变化
            betweenness, degrees, second_order, third_order = calculate_metrics(G_step)
            data.append((betweenness, degrees, second_order, third_order))
        """介数曲线绘制"""
        df_betweenness = pd.DataFrame([d[0] for d in data])
        df_betweenness.index = range(1, len(df_betweenness) + 1)
        # try:
        #     df_betweenness.plot(ax=ax,legend=False,cmap=custom_colormap,alpha=0.9,linewidth=1)
        # except:
        #     print('error')
        df_betweenness.fillna(0, inplace=True)
        """对瓦解过程的介数曲线进行聚类"""
        df =df_betweenness.T
        df.fillna(0, inplace=True)
        scaled_data = df.values
        from tslearn.clustering import TimeSeriesKMeans, silhouette_score
        # 将数据转换为 tslearn 能接受的格式
        formatted_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
        metric='euclidean'
        best_score, best_k, best_model = -1, None, None
        inertias = []
        k1, k2 = 3, 7
        for k in range(k1, k2+1):
            model = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=10,random_state=0)
            labels = model.fit_predict(formatted_data)
            score = silhouette_score(formatted_data, labels, metric=metric)
            inertias.append(model.inertia_)
            if score > best_score:
                best_score = score
                best_k = k
                best_model = model
        # print(f"Optimal number of clusters: {best_k}")
        """按照峰值出现时间由前到后排序类别"""
        labels = list(set(best_model.labels_))
        cluster_dict = {label: [] for label in labels}
        for node, label in zip(df.index, best_model.labels_):
            cluster_dict[label].append(node)
        time_with_cluster = [[list(best_model.cluster_centers_[x].ravel()).index(max(best_model.cluster_centers_[x].ravel())),cluster_dict[x]] for x in cluster_dict.keys()]  # 每个cluster的峰值
        time_with_cluster = sorted(time_with_cluster,key=lambda x:x[0])
        zero_time = []
        for i in range(len(time_with_cluster)):
            if time_with_cluster[i][0]==0:
                zero_time+=time_with_cluster[i][1]
        new_time_with_cluster = [[0,zero_time]]+[x for x in time_with_cluster if x[0]!=0] if zero_time!=[] else time_with_cluster
        first_time = new_time_with_cluster[1][0]    # 第一个骨干出现时间***************************************
        first_peak_data_list.append(first_time)
    return first_peak_data_list

if __name__=='__main__':
    outpath = "./backbone_result/N100/statistic/" # 输出路径
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    plt.rcParams['font.family'] = 'Arial'  # 设置字体
    plt.rcParams['font.size'] = 16  # 设置字号
    colors = color_get()
    # colors = [colors[i] for i in [0,3,1,4,6,5]]
    attack='HDA'
    graph_type = 'BA'
    n=100
    sequence = statisics(graph_type,attack,n)
    sequence = [x/n for x in sequence]
    fig, ax = plt.subplots(figsize=(12, 12), sharex=True, sharey=True)
    method = 'RL'
    # 绘制每个数据集的直方图
    ax.hist(sequence, color=colors[0], alpha=0.7, label="%s" % (method), edgecolor='black',linewidth=0.5)
    ax.axvline(x=sum(sequence) / len(sequence), color='black', linestyle='--', label='mean',linewidth=1.1)
    ax.legend()
    major_locator = MultipleLocator(base=20)  # 主刻度位置间隔
    minor_locator = AutoMinorLocator(n=1)  # 次刻度数量
    ax.yaxis.set_major_locator(major_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    major_locator = MultipleLocator(base=0.1)  # 主刻度位置间隔
    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
    ax.tick_params(axis='both', which='major', length=6, width=1)  # 主刻度线
    ax.tick_params(axis='both', which='minor', length=3, width=1)  # 次刻度线
    plt.xlim(0,0.3)
    plt.ylim(0,45)
    plt.xlabel('Fracton of Nodes Removed')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(outpath+'First peak moment distribution %s_%s_%s.jpg'%(attack,graph_type,n))
    plt.show()
    print('Finish')
