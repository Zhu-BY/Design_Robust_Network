import networkx as nx
import torch
from pyvis.network import Network
import numpy as np
from backbone_for_100 import draw_dismanting_curve
import matplotlib.pyplot as plt
import copy
import pandas as pd
from backbone_utils import color_get,calculate_metrics
from matplotlib.colors import LinearSegmentedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_nx_with_cluster(G,path,nodes_cluster=dict()):
    G_nodes = list(G.nodes())
    betweenness = nx.betweenness_centrality(G)
    degrees = [G.degree(n) for n in G_nodes]
    sizes = dict()
    for node in G_nodes:
        # sizes[node] = (G.degree(node)+1) * 20
        sizes[node] = (betweenness[node]) * 350+20
    # 可视化
    colors = color_get()
    if len(set(list(nodes_cluster.values()))) == 3:
        rgba_colors = [colors[0], colors[1], colors[6]]
    elif len(set(list(nodes_cluster.values()))) == 4:
        rgba_colors = [colors[0], colors[1], colors[3], colors[-1]]
    elif len(set(list(nodes_cluster.values()))) == 5:
        rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[-1]]
    elif len(set(list(nodes_cluster.values()))) == 6:
        rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[-1]]
    elif len(set(list(nodes_cluster.values()))) == 7:
        rgba_colors = colors
    else:
        rgba_colors=colors
    rgba_colors.reverse()
    hex_colors = ['#%02x%02x%02x' % (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)) for
                  rgba_color in rgba_colors]

    colors=dict()
    edgecolors=dict()
    linewidths = dict()
    for node in sizes.keys():
        cluster_id =nodes_cluster[node]
        node_color = hex_colors[cluster_id % len(hex_colors)]
        edgecolor = 'gray'
        linewidths[node]=0.3
        colors[node] = node_color
        edgecolors[node]=edgecolor
    edge_colors = dict()
    widths = dict()
    for edge in G.edges():
        if nodes_cluster[edge[0]]==nodes_cluster[edge[1]]:
            edge_colors[edge] = hex_colors[nodes_cluster[edge[0]] % len(hex_colors)]
            widths[edge] = 1.2
        else:
            edge_colors[edge] = 'lightgray'
            widths[edge] = 0.8
    # plt.figure(figsize=(2.8,2.8))
    # nx.draw(G,pos,nodelist =list(sizes.keys()),node_size=list(sizes.values()),node_color=list(colors.values()),edgelist = list(edge_colors.keys()),
    #         linewidths=list(linewidths.values()),edge_color=list(edge_colors.values()),width =list(widths.values()) ,edgecolors=list(edgecolors.values()))
    # plt.suptitle("%s" %step,fontsize=12)
    # plt.savefig(f"%s"%path+'%s.jpg'%pos_name,dpi=300)
    # plt.show()
    # 绘制pyvis
    sizes = dict()
    for node in G_nodes:
        sizes[node] = (G.degree(node)+1) * 5
    net = Network(notebook=True, height="750px", width="100%")
    net.toggle_physics(False)
    for node in sizes.keys():
        cluster_id =nodes_cluster[node]
        node_color = hex_colors[cluster_id % len(hex_colors)]
        edgecolor = 'gray'
        linewidths[node]=0.3
        colors[node] = node_color
        edgecolors[node]=edgecolor
        net.add_node(node, color=node_color, size=sizes[node],border_color=edgecolors[node], border_width=linewidths[node])
    # 添加边
    for edge in G.edges():
        if nodes_cluster[edge[0]]==nodes_cluster[edge[1]]:
            edge_colors[edge] = hex_colors[nodes_cluster[edge[0]] % len(hex_colors)]
            widths[edge] = 6
        else:
            edge_colors[edge] = 'lightgray'
            widths[edge] = 2
        source, target = edge
        net.add_edge(int(source), int(target), width=widths[edge], color=edge_colors[edge])
    # 设置position
    if len(G)==600:
        import json
        with open("./backbone_result/N600/"+'position_shown_backbone.json', 'r') as f:
            positions = json.load(f)
        for node_id, pos in positions.items():
            net.get_node(int(node_id))['x'] = pos['x']
            net.get_node(int(node_id))['y'] = pos['y']
    # 设置结束
    net.show_buttons(filter_=['physics'])
    net.show(f"%s.html"%path)
    print(' ')
    # var positions = network.getPositions();

def show_case():
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    plt.rcParams['font.family'] = 'Arial'  # 设置字体
    plt.rcParams['font.size'] = 16  # 设置字号
    colors = color_get()
    colors.reverse()
    custom_colormap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=10)
    graph_types = ['BA']
    attack_types =['HDA']
    ns = [600]  # 网络规模
    for n in ns:
        for graph_type in graph_types: # 对于每一类网络进行设计
            for attack in attack_types: # 对于每一种攻击
                out_path = "./backbone_result/N%s/%s/"%(n,attack)
                data_loaded = torch.load('../RL_Algorithm/Design_result/synthetic_result/N%s/%s_%s_%s_rl_result.pth'%(n,attack,graph_type,n))
                G_list = data_loaded[0][-1]
                dismantle_number= 1 if attack not in ['GND', 'GNDR'] else 4
                """设计过程三个网络瓦解曲线"""
                G_show_list = [G_list[0],G_list[50],G_list[-1]]
                G0 = G_list[0]
                name_list = ['Initial','Step 50','Step 101']
                R_list_now, remove_nodes_list = draw_dismanting_curve(out_path + '0', graph_type, attack, G_list[0],G_show_list,name_list)
                for G, name,remove_nodes, R_G in zip(G_show_list,name_list,remove_nodes_list[1:],R_list_now[1:]):
                    """瓦解过程特征统计"""
                    G_step = copy.deepcopy(G)
                    betweenness, degrees, second_order, third_order = calculate_metrics(G_step)
                    data = []
                    # step_num= 60
                    for step in range(10000):
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
                        if len(gcc)<0.1*len(G):
                            break
                    """介数曲线绘制"""
                    df_betweenness = pd.DataFrame([d[0] for d in data])
                    df_betweenness.index = range(1, len(df_betweenness) + 1)
                    fig, ax = plt.subplots(1, 1, figsize=(7, 2))
                    # 介数中心性
                    df_betweenness.plot(ax=ax,legend=False,cmap=custom_colormap,alpha=0.9,linewidth=1)
                    df_betweenness.fillna(0, inplace=True)
                    plt.xlim(0,45)
                    # plt.ylim(-0.01,np.max(df_betweenness.values)+0.01)
                    # ylim = np.max(df_betweenness.values)+0.01
                    ylim =0.2
                    plt.ylim(-0.01,ylim)
                    # # 设置图例
                    # from matplotlib.ticker import MultipleLocator, AutoMinorLocator
                    # legend = ax.legend(frameon=1, fontsize=12, loc='upper right')
                    # frame = legend.get_frame()
                    # frame.set_color('none')  # 设置图例边框颜色
                    # frame.set_alpha(0)  # 设置图例边框透明
                    # 设置y轴刻度数量
                    major_locator = MultipleLocator(base=0.2)# 主刻度位置间隔
                    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                    plt.gca().yaxis.set_major_locator(major_locator)
                    plt.gca().yaxis.set_minor_locator(minor_locator)
                    # 设置x轴刻度数量
                    major_locator = MultipleLocator(base=20)  # 主刻度位置间隔
                    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                    plt.gca().xaxis.set_major_locator(major_locator)
                    plt.gca().xaxis.set_minor_locator(minor_locator)
                    # 设置刻度样式
                    plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
                    plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
                    plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
                    plt.tight_layout()
                    plt.savefig(out_path+'Betweenness_Centrality %s_%s_%s.jpg'%(graph_type,attack,name),dpi=300)
                    plt.show()
                    """对瓦解过程的介数曲线进行聚类"""
                    df =df_betweenness.T
                    df.fillna(0, inplace=True)
                    scaled_data = df.values
                    from tslearn.clustering import TimeSeriesKMeans, silhouette_score
                    # 将数据转换为 tslearn 能接受的格式
                    formatted_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
                    for metric in ['euclidean']:
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
                        print(f"Optimal number of clusters: {best_k}")
                        """按照峰值出现时间由前到后排序类别"""
                        labels = list(set(best_model.labels_))
                        cluster_dict = {label: [] for label in labels}
                        for node, label in zip(df.index, best_model.labels_):
                            cluster_dict[label].append(node)
                        label_order=sorted(list(cluster_dict.keys()),key=lambda x:list(best_model.cluster_centers_[x].ravel()).index(max(best_model.cluster_centers_[x].ravel())))
                        """按照峰值出现时间排序，并把峰值出现时间为0的cluster放在一起"""
                        time_with_cluster0 = [[list(best_model.cluster_centers_[x].ravel()).index(
                            max(best_model.cluster_centers_[x].ravel())), cluster_dict[x]] for x in
                                             cluster_dict.keys()]  # 每个cluster的峰值
                        zero_num = len([x for x in time_with_cluster0 if x[0]==0]) # 峰值为0的数量
                        time_with_cluster = sorted(time_with_cluster0, key=lambda x: x[0])
                        zero_time = []
                        for i in range(len(time_with_cluster)):
                            if time_with_cluster[i][0] == 0:
                                zero_time += time_with_cluster[i][1]
                        new_time_with_cluster = [[0, zero_time]] + [x for x in time_with_cluster if x[0] != 0] if zero_time != [] else time_with_cluster
                        new_cluster_dict = {i: new_time_with_cluster[i][1] for i in range(len(new_time_with_cluster))} # 新的聚类结果
                        # 创建node-label字典
                        node_cluster_dict = dict()
                        for key in new_cluster_dict.keys():
                            node_l = new_cluster_dict[key]
                            for node in node_l:
                                node_cluster_dict[node] = key
                        for node in G.nodes():
                            if node not in node_cluster_dict.keys():
                                node_cluster_dict[node] = 0
                        # draw_community_nx_with_cluster(G, path + '%s %s %s %s' % (name, 0, 0, attack), 'step 0: 100',pos, nodes_cluster=node_cluster_dict)
                        """导入颜色"""
                        colors = color_get()
                        if len(cluster_dict) == 3:
                            rgba_colors = [colors[0], colors[1], colors[6]]
                        if len(cluster_dict) == 4:
                            rgba_colors = [colors[0], colors[1], colors[3], colors[-1]]
                        if len(cluster_dict) == 5:
                            rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[-1]]
                        if len(cluster_dict) == 6:
                            rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[-1]]
                        if len(cluster_dict) == 7:
                            rgba_colors = colors
                        rgba_colors.reverse()
                        """可视化聚类结果"""
                        for yi,label in zip(range(best_k),label_order):
                            fig,ax = plt.subplots(figsize=(7, 2))
                            for xx in formatted_data[best_model.labels_ == label]:
                                x = list(range(1,len(xx)+1))
                                x = [y/len(G) for y in x]
                                ax.plot(x,xx.ravel(), "gray", alpha=0.2)
                            if time_with_cluster0[label][0]==0:
                                ax.plot(x,best_model.cluster_centers_[label].ravel(), "-",color = rgba_colors[0],linewidth=3,alpha=1,label =f"Cluster {yi + 1}")
                            else:
                                ax.plot(x, best_model.cluster_centers_[label].ravel(), "-", color=rgba_colors[yi-max(0,(zero_num-1))], linewidth=3, alpha=1, label=f"Cluster {yi + 1}")
                            plt.xlim(0, 45/len(G))
                            plt.ylim((-0.01,ylim))
                            # 设置图例
                            from matplotlib.ticker import MultipleLocator, AutoMinorLocator
                            legend = ax.legend(frameon=1, fontsize=12, loc='upper right')
                            frame = legend.get_frame()
                            frame.set_color('none')  # 设置图例边框颜色
                            frame.set_alpha(0)  # 设置图例边框透明
                            # 设置y轴刻度数量
                            major_locator = MultipleLocator(base=0.2) # 主刻度位置间隔
                            minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                            plt.gca().yaxis.set_major_locator(major_locator)
                            plt.gca().yaxis.set_minor_locator(minor_locator)
                            # 设置x轴刻度数量
                            major_locator = MultipleLocator(base=0.03)  # 主刻度位置间隔
                            minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                            plt.gca().xaxis.set_major_locator(major_locator)
                            plt.gca().xaxis.set_minor_locator(minor_locator)
                            # 设置刻度样式
                            plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
                            plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
                            plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
                            plt.tight_layout()
                            # plt.title('cluster score: %s'%best_score)
                            plt.tight_layout()
                            plt.savefig(out_path+'Cluster_%s %s %s %s %s.jpg'%(yi+1,graph_type,attack,name,metric),dpi=300)
                            plt.show()

                        """取每一类节点的最大连通子图"""
                        cluster_dict = new_cluster_dict
                        for cid in cluster_dict.keys():
                            cluster = cluster_dict[cid]
                            G_ = copy.deepcopy(G)
                            backbone_G = G_.subgraph(cluster)
                            gcc_cluster = list(max(list(nx.connected_components(backbone_G)), key=len))
                            cluster_dict[cid] = gcc_cluster
                        # 更换node_cluster_dict标签
                        cluster_dict_new = cluster_dict
                        for node in node_cluster_dict.keys():
                            zero_cluster = True
                            for key in cluster_dict_new.keys():
                                if node in cluster_dict_new[key]:
                                    node_cluster_dict[node] = key
                                    zero_cluster = False
                                    break
                            if zero_cluster:
                                cluster_dict_new[0].append(node)
                                node_cluster_dict[node] = 0
                        """绘制瓦解过程不同cluster节点介数整体的变化值"""
                        cluster_labels = set(list(node_cluster_dict.values()))
                        cluster_bet_delta = dict()
                        for label in cluster_labels:
                            cluster_bet_delta[label] = [0]
                        Gcc_list = [1]
                        G_step = copy.deepcopy(G)
                        bet_last = nx.betweenness_centrality(G_step)
                        for step in range(len(data)):
                            G_step.remove_node(remove_nodes[step])
                            bet_now = nx.betweenness_centrality(G_step)
                            gcc = list(max(nx.connected_components(G_step), key=len))
                            Gcc_list.append(len(gcc) / len(G))
                            for cluster in cluster_dict_new.keys():
                                node_list = cluster_dict_new[cluster]
                                delta_avg_bet = (sum(bet_now[node] for node in node_list if node in G_step.nodes())-sum(bet_last[node] for node in node_list if node in G_step.nodes()))/len(node_list) # 介数的平均变化
                                cluster_bet_delta[cluster].append(delta_avg_bet)
                            bet_last = bet_now
                        color = 'black'
                        fig, ax1 = plt.subplots(figsize=(6.9,6))
                        ax1.set_xlabel('Fraction of Nodes Removed')
                        ax1.set_ylabel('Gcc size')
                        # 瓦解曲线
                        x = np.linspace(0, len(Gcc_list) - 1, len(Gcc_list))
                        x = [i / len(G) for i in x]
                        ax1.plot(x, Gcc_list, color=color, linewidth=1, alpha=0.5,zorder=3)
                        ax1.scatter(x, Gcc_list, marker='o', s=30, alpha=0.5, c=color, edgecolors=color,
                                    label='%s: %.2f' % (name, R_G),zorder=3)
                        # cluster节点移除比例
                        ax2 = ax1.twinx()
                        ax2.set_ylabel('Delta of Betweenness')
                        for cluster, color in zip(cluster_bet_delta.keys(), rgba_colors):
                            x = np.linspace(0, len(cluster_bet_delta[cluster]) - 1, len(cluster_bet_delta[cluster])) #
                            x = [i / len(G0) for i in x]
                            normalized_data_np = cluster_bet_delta[cluster]
                            # ax2.plot(x, cluster_bet_delta[cluster], color=color, alpha=0.5,linewidth=1)
                            ax2.plot(x, normalized_data_np, color=color, alpha=0.8,linewidth=1)
                            ax2.scatter(x, normalized_data_np, color=color, s=12, alpha=0.99,
                                        label='potential backbone %s' % (cluster) if cluster>0 else 'other nodes')
                        legend = ax1.legend(frameon=1, bbox_to_anchor=(0.8, 0.7), fontsize=10) if name=="Initial" else ax1.legend(frameon=1, bbox_to_anchor=(0.4, 0.1), fontsize=10)
                        frame = legend.get_frame()
                        frame.set_color('none')  # 设置图例边框颜色
                        frame.set_alpha(0)  # 设置图例边框透明
                        legend = ax2.legend(frameon=1, bbox_to_anchor=(0.8, 0.8), fontsize=10) if name=="Initial" else ax2.legend(frameon=1, bbox_to_anchor=(0.4, 0.4), fontsize=10)
                        frame = legend.get_frame()
                        frame.set_color('none')  # 设置图例边框颜色
                        frame.set_alpha(0)  # 设置图例边框透明
                        # 设置y1轴刻度数量
                        major_locator = MultipleLocator(base=0.5) # 主刻度位置间隔
                        minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                        ax1.yaxis.set_major_locator(major_locator)
                        ax1.yaxis.set_minor_locator(minor_locator)
                        # 设置y2轴刻度数量
                        # ymax = max([max(l) for l in cluster_bet_delta.values()])
                        # ymin = min([min(l) for l in cluster_bet_delta.values()])
                        # major_locator = MultipleLocator(base=0.1) if name=="Initial" else MultipleLocator(base=0.02)
                        major_locator = MultipleLocator(base=0.02)
                        minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                        ax2.yaxis.set_major_locator(major_locator)
                        ax2.yaxis.set_minor_locator(minor_locator)
                        # if name=='Initial':
                        #     ax2.set_ylim(-0.4,0.01)
                        # else:
                        ax2.set_ylim(-0.05,0.03)
                        # 设置x轴刻度数量
                        major_locator = MultipleLocator(base=0.03)  # 主刻度位置间隔
                        minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                        plt.gca().xaxis.set_major_locator(major_locator)
                        plt.gca().xaxis.set_minor_locator(minor_locator)
                        # 设置刻度样式
                        plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
                        plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
                        plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
                        # plt.title('Dela')  # Size of Largest Connected Component
                        xmax = (len(data)+1) / len(G0)
                        plt.subplots_adjust(right=0.8)
                        # plt.xlim(-0.01, 0.1)
                        # plt.xlim(-0.01, xmax)
                        plt.xlim(-0.01, 0.075)
                        fig.patch.set_alpha(0)
                        ax1.patch.set_alpha(0)
                        ax2.patch.set_alpha(0)
                        plt.savefig(out_path + 'Delta_betweenness %s %s %s.png' % (graph_type, attack, name), dpi=300)
                        plt.show()
                        """瓦解过程中不同cluster节点标注"""
                        draw_nx_with_cluster(G, out_path + 'Backbones %s %s %s' % (graph_type,attack,name), nodes_cluster=node_cluster_dict)

if __name__=='__main__':
    show_case()