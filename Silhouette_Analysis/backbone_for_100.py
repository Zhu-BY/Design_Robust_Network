import pandas as pd
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from RL_Algorithm.Environment.dismantlers.dismantlers_with_seeds import dismantle_with_seeds
import torch
import numpy as np
from collections import Counter
from pyvis.network import Network
import networkx as nx
import matplotlib.pyplot as plt
import copy
from backbone_utils import generate_network,calculate_metrics,color_get
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_dismanting_curve(path,type,dismantling_name,G0, G_list, name_list,r_curve_list=0,x_max0=0):
    tab20c = plt.cm.get_cmap('tab20c', 20)
    colors1 = [tab20c(4), tab20c(8), tab20c(12),tab20c(0),tab20c(16)]
    colors1.reverse()
    N=len(G0)
    if r_curve_list==0:
        name_list = ['initial']+name_list
        N=len(G0)
        r_curve_list = []
        remove_nodes_list = []
        for G in [G0]+G_list:
            dismantling_number = 1 if dismantling_name not in ['GND','GNDR'] else 4
            r, curve,m,remove_nodes = dismantle_with_seeds(dismantling_name,G, dismantling_number)
            r_curve_list.append([r,curve])
            remove_nodes_list.append(remove_nodes)

    fig, ax = plt.subplots(figsize=(6, 6))
    # fig, ax = plt.subplots()
    for plot,color1 in zip(r_curve_list,colors1):
        x = np.linspace(0, len(plot[1]) - 1, len(plot[1]))
        x = [i/N for i in x]
        ax.plot(x,plot[1], color=color1, linewidth=1,alpha=0.5)
    for plot,name,color1 in zip(r_curve_list,name_list,colors1):
        x = np.linspace(0, len(plot[1]) - 1, len(plot[1]))
        x = [i/N for i in x]
        ax.scatter(x,plot[1], marker='o', s=30, alpha=0.5, color=color1, edgecolors=color1, label='%s: %.2f' % (name,plot[0]))

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
    # plt.savefig(path+"dismantling %s %s.png"%(type,dismantling_name), dpi=600)
    plt.show()
    return [x[0] for x in r_curve_list],remove_nodes_list # 返回R
def draw_nx_with_cluster(G,path,nodes_cluster=dict(),num=0):
    G_nodes = list(G.nodes())
    betweenness = nx.betweenness_centrality(G)
    degrees = [G.degree(n) for n in G_nodes]
    sizes = dict()
    for node in G_nodes:
        # sizes[node] = (G.degree(node)+1) * 20
        # sizes[node] = (betweenness[node]) * 350+20
        sizes[node] = 1.5 * 20
    # 可视化
    colors = color_get()
    if num == 3:
        rgba_colors = [colors[0], colors[1], colors[6]]
    if num == 4:
        rgba_colors = [colors[0], colors[1], colors[3], colors[-1]]
    if num == 5:
        rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[-1]]
    if num == 6:
        rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[4], colors[-1]]
    if num == 7:
        rgba_colors = colors

    # rgba_colors = rgba_colors[0:(len(set(list(nodes_cluster.values())))- 1)] + rgba_colors[-1:]
    rgba_colors.reverse()
    hex_colors = ['#%02x%02x%02x' % (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)) for
                  rgba_color in rgba_colors]
    # cmap_ = plt.get_cmap('tab10')
    # rgba_colors = [cmap_(1),cmap_(3),cmap_(0),cmap_(2),cmap_(7)]  # 红 蓝 绿 灰
    # hex_colors = ['#%02x%02x%02x' % (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)) for
    #               rgba_color in rgba_colors]

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
            widths[edge] = 1.9
        else:
            edge_colors[edge] = 'lightgray'
            widths[edge] = 0.8
    # 绘制pyvis
    sizes = dict()
    for node in G_nodes:
        # sizes[node] = (G.degree(node)+1) * 5
        sizes[node] = 20
    net = Network(notebook=True, height="750px", width="100%")
    for node in sizes.keys():
        cluster_id =nodes_cluster[node]
        node_color = hex_colors[cluster_id % len(hex_colors)]
        edgecolor = 'gray'
        linewidths[node]=1
        colors[node] = node_color
        edgecolors[node]=edgecolor
        net.add_node(node, color=node_color, size=sizes[node],border_color=edgecolors[node], border_width=linewidths[node])

    # 添加边
    for edge in G.edges():
        if nodes_cluster[edge[0]]==nodes_cluster[edge[1]]:
            edge_colors[edge] = hex_colors[nodes_cluster[edge[0]] % len(hex_colors)]
            if nodes_cluster[edge[0]]!=0:
                widths[edge] = 25
            else:
                widths[edge]=6

        else:
            edge_colors[edge] = 'lightgray'
            widths[edge] = 6
        source, target = edge
        net.add_edge(int(source), int(target), width=widths[edge], color=edge_colors[edge])
    net.show_buttons(filter_=['physics'])
    net.show(f"%s.html"%path)
    print(' ')
    # var positions = network.getPositions();
def Silhouette_Analysis(show_case_id = 50):
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    plt.rcParams['font.family'] = 'Arial'  # 设置字体
    plt.rcParams['font.size'] = 16  # 设置字号
    colors = color_get()
    from matplotlib.colors import LinearSegmentedColormap
    colors.reverse()
    custom_colormap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=10)
    # 载入数据
    graph_type='BA'
    n=100
    attacks = ['HDA', 'HBA', 'CI2', 'MS', 'GND', 'GNDR']
    ydict = {'HDA':[0.25,-0.15], 'HBA':[0.25,-0.3], 'CI2':[0.2,-0.25], 'MS':[0.05,-0.1], 'GND':[0.25,-0.35], 'GNDR':[0.1,-0.2]}
    for attack in attacks:
        path = "./backbone_result/N%s/%s/"%(n,attack) # 输出路径
        G_Rl = torch.load('../RL_Algorithm/Design_result/synthetic_result/N%s/%s_%s_%s_rl_result.pth'%(n,attack,graph_type,n))[show_case_id][2][-1]
        G0, _ = generate_network(n, graph_type, show_case_id)
        """强化学习比DCBO的度提升节点在社团分解中的作用"""
        R_list_now, remove_nodes_list = draw_dismanting_curve(path + '%s' % show_case_id, graph_type, attack, G0, [G_Rl],['RL'])
        for G, name,remove_nodes in zip([G_Rl], ['RL'],remove_nodes_list[1:]):
            G_step = copy.deepcopy(G)
            # betweenness, degrees, second_order, third_order = calculate_metrics(G_step)
            data = []
            # data.append((betweenness, degrees, second_order, third_order))
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
                betweenness,_,_,_ = calculate_metrics(G_step)
                data.append(betweenness)
            """介数曲线绘制"""
            df_betweenness = pd.DataFrame(data)
            df_betweenness.index = range(1, len(df_betweenness) + 1)
            fig, ax = plt.subplots(1, 1, figsize=(7, 2))
            # 介数中心性
            df_betweenness.plot(ax=ax,legend=False,cmap=custom_colormap,alpha=0.9,linewidth=1)
            df_betweenness.fillna(0, inplace=True)
            plt.xlim(0,len(data)+1)
            plt.ylim(-0.01,np.max(df_betweenness.values)+0.01)
            ylim = np.max(df_betweenness.values)+0.01
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
            plt.savefig(path+'Betweenness_Centrality %s_%s_%s.jpg'%(graph_type,attack,name),dpi=300)
            plt.show()
            """对瓦解过程的介数曲线进行聚类"""
            df =df_betweenness.T
            df.fillna(0, inplace=True)
            scaled_data = df.values
            from tslearn.clustering import TimeSeriesKMeans, silhouette_score
            formatted_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
            for metric in ['euclidean']:
                best_score,best_k,best_model = -1,None,None
                inertias = []
                k1,k2 = 3,7
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
                label_order = sorted(list(cluster_dict.keys()),key=lambda x: list(best_model.cluster_centers_[x].ravel()).index(max(best_model.cluster_centers_[x].ravel())))
                """按照峰值出现时间排序，并把峰值出现时间为0的cluster放在一起"""
                time_with_cluster0 = [[list(best_model.cluster_centers_[x].ravel()).index(
                    max(best_model.cluster_centers_[x].ravel())), cluster_dict[x]] for x in
                    cluster_dict.keys()]  # 每个cluster的峰值
                zero_num = len([x for x in time_with_cluster0 if x[0] == 0])  # 峰值为0的数量
                time_with_cluster = sorted(time_with_cluster0, key=lambda x: x[0])
                zero_time = []
                for i in range(len(time_with_cluster)):
                    if time_with_cluster[i][0] == 0:
                        zero_time += time_with_cluster[i][1]
                new_time_with_cluster = [[0, zero_time]] + [x for x in time_with_cluster if x[0] != 0] if zero_time != [] else time_with_cluster
                new_cluster_dict = {i: new_time_with_cluster[i][1] for i in range(len(new_time_with_cluster))}  # 新的聚类结果
                # 创建node-label字典
                node_cluster_dict = dict()
                for key in new_cluster_dict.keys():
                    node_l = new_cluster_dict[key]
                    for node in node_l:
                        node_cluster_dict[node] = key
                for node in G.nodes():
                    if node not in node_cluster_dict.keys():
                        node_cluster_dict[node] = 0
                """导入颜色"""
                colors = color_get()
                if len(cluster_dict) == 3:
                    rgba_colors = [colors[0], colors[1], colors[6]]
                if len(cluster_dict) == 4:
                    rgba_colors = [colors[0], colors[1], colors[3], colors[-1]]
                if len(cluster_dict) == 5:
                    rgba_colors = [colors[0], colors[1], colors[2],colors[3], colors[-1]]
                if len(cluster_dict) == 6:
                    rgba_colors = [colors[0], colors[1], colors[2],colors[3], colors[4],colors[-1]]
                if len(cluster_dict) == 7:
                    rgba_colors = colors
                rgba_colors.reverse()
                """可视化聚类结果"""
                for yi,label in zip(range(best_k),label_order):
                    fig,ax = plt.subplots(figsize=(7, 2))
                    for xx in formatted_data[best_model.labels_ == label]:
                        x = list(range(1,len(xx)+1))
                        x = [e/len(G) for e in x]
                        ax.plot(x,xx.ravel(), "gray", alpha=0.2)
                    if time_with_cluster0[label][0] == 0:
                        ax.plot(x, best_model.cluster_centers_[label].ravel(), "-", color=rgba_colors[0],
                                linewidth=3, alpha=1, label=f"Cluster {yi + 1}")
                    else:
                        ax.plot(x, best_model.cluster_centers_[label].ravel(), "-",
                                color=rgba_colors[yi - max(0, (zero_num - 1))], linewidth=3, alpha=1,
                                label=f"Cluster {yi + 1}")
                    plt.xlim(-0.01/len(G), (scaled_data.shape[1]+1)/len(G))
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
                    major_locator = MultipleLocator(base=20/len(G))  # 主刻度位置间隔
                    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                    plt.gca().xaxis.set_major_locator(major_locator)
                    plt.gca().xaxis.set_minor_locator(minor_locator)
                    # plt.xlabel('Fraction of Nodes Removed')
                    # 设置刻度样式
                    plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
                    plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
                    plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
                    plt.tight_layout()
                    # plt.title('cluster score: %s'%best_score)
                    plt.tight_layout()
                    plt.savefig(path+'Cluster_%s %s %s %s %s.jpg'%(yi+1,graph_type,attack,name,metric),dpi=300)
                    plt.show()

                """###################取每一类节点的最大连通子图 （规模要求大于0）#####################"""
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
                cluster_labels = set(list(cluster_dict_new.keys()))
                cluster_bet_delta = dict()
                for label in cluster_labels:
                    cluster_bet_delta[label] = [0]  # 初始为0
                Gcc_list = [1]
                G_step = copy.deepcopy(G)
                bet_last = nx.betweenness_centrality(G_step)
                step_num = 40 if attack != 'MS' else 60
                for step in range(step_num):
                    G_step.remove_node(remove_nodes[step])
                    bet_now = nx.betweenness_centrality(G_step)
                    gcc = list(max(nx.connected_components(G_step), key=len))
                    if len(gcc)<=2:
                        break
                    Gcc_list.append(len(gcc) / len(G))
                    for cluster in cluster_dict_new.keys():
                        node_list = cluster_dict_new[cluster]
                        try:
                            delta_avg_bet = (sum(bet_now[node] for node in node_list if node in G_step.nodes())-sum(bet_last[node] for node in node_list if node in G_step.nodes()))/len(node_list) # 介数的平均变化
                            cluster_bet_delta[cluster].append(delta_avg_bet)
                        except:
                            cluster_bet_delta[cluster]=0
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
                            label='%s: %.2f' % (name, sum(Gcc_list)),zorder=3)
                # cluster节点移除比例
                ax2 = ax1.twinx()
                ax2.set_ylabel('Delta of Betweenness')
                n_b=0
                for cluster, color in zip(cluster_bet_delta.keys(), rgba_colors):
                    if cluster_bet_delta[cluster]!=0:
                        x = np.linspace(0, len(cluster_bet_delta[cluster]) - 1, len(cluster_bet_delta[cluster]))
                        x = [i / len(G0) for i in x]
                        normalized_data_np = cluster_bet_delta[cluster]
                        # ax2.plot(x, cluster_bet_delta[cluster], color=color, alpha=0.5,linewidth=1)
                        ax2.plot(x, normalized_data_np, color=color, alpha=0.8,linewidth=1)
                        ax2.scatter(x, normalized_data_np, color=color, s=12, alpha=0.99,
                                    label='backbone %s' % (n_b) if n_b>0 else 'other nodes')
                        n_b=n_b+1
                legend = ax1.legend(frameon=1, loc='upper right', bbox_to_anchor=(0.35, 0.45), fontsize=10) if attack in ['GND','GNDR'] else ax1.legend(frameon=1, loc='upper right', bbox_to_anchor=(1, 0.8), fontsize=10)
                frame = legend.get_frame()
                frame.set_color('none')  # 设置图例边框颜色
                frame.set_alpha(0)  # 设置图例边框透明
                legend = ax2.legend(frameon=1,loc='upper right', bbox_to_anchor=(0.35, 0.4), fontsize=10)if attack in ['GND','GNDR']  and name=='RL' else ax2.legend(frameon=1,loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
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
                ymax = ydict[attack][0]
                ymin = ydict[attack][1]
                major_locator = MultipleLocator(base=(ymax-ymin)/4) # 主刻度位置间隔
                minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                ax2.yaxis.set_major_locator(major_locator)
                ax2.yaxis.set_minor_locator(minor_locator)
                # 设置x轴刻度数量
                major_locator = MultipleLocator(base=0.2)  # 主刻度位置间隔
                minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                plt.gca().xaxis.set_major_locator(major_locator)
                plt.gca().xaxis.set_minor_locator(minor_locator)
                # 设置刻度样式
                plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
                plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
                plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
                # plt.title('Dela')  # Size of Largest Connected Component
                xmax = max([len(x) for x in remove_nodes_list]) / len(G0)
                plt.subplots_adjust(right=0.8)
                plt.xlim(-0.01, (step_num+1)/len(G))
                plt.ylim(ymin,ymax)
                fig.patch.set_alpha(0)
                ax1.patch.set_alpha(0)
                ax2.patch.set_alpha(0)
                plt.savefig(path + 'Delta_betweenness %s %s %s.png' % (graph_type, attack, name), dpi=300)
                plt.show()
                """瓦解过程中不同cluster节点标注"""
                draw_nx_with_cluster(G, path + 'Backbones %s %s %s' % (graph_type,attack,name), nodes_cluster=node_cluster_dict,num=len(cluster_dict_new))
                """绘制依次移除的分组的渗流图"""
                if attack=='HDA':
                    import itertools
                    permutations = list(itertools.permutations(list(cluster_dict_new.keys())))
                    permutations = [permutations[0],permutations[-1]]
                    for perm in permutations:
                        Gcc_list = dict()
                        G_step = copy.deepcopy(G)
                        # perm=[0,1,2]
                        for number in perm:
                            Gcc_list[number] = []
                            node_list = copy.deepcopy(cluster_dict_new[number])
                            while len(node_list)!=0 and len(G_step)>0:
                                deg_list = [G.degree(x) for x in node_list]
                                idx = deg_list.index(max(deg_list))
                                node  = node_list[idx]
                                node_list.pop(idx)
                                G_step.remove_node(node)
                                try:
                                    gcc = list(max(nx.connected_components(G_step), key=len))
                                except:
                                    gcc=[]
                                Gcc_list[number].append(len(gcc) / len(G))
                        Gcc_list_list = list(Gcc_list.values())
                        Gcc_list_len = [0]+[len(x) for x in Gcc_list_list]
                        xs = np.linspace(0, sum(Gcc_list_len) - 1, sum(Gcc_list_len))
                        xs = [i / len(G0) for i in xs]

                        fig, ax = plt.subplots(figsize=(6,6))
                        for i in range(len(Gcc_list_list)):
                            color = rgba_colors[perm[i]]
                            x = xs[sum([Gcc_list_len[j] for j in range(0,i+1)]):sum([Gcc_list_len[j] for j in range(0,i+2)])]
                            ax.plot(x, Gcc_list_list[i], color=color, alpha=0.5)
                            ax.scatter(x,Gcc_list_list[i], color=color, s=12, alpha=0.8,
                                        label='cluster %s' % (cluster + 1))
                        plt.xlabel('Fraction of Nodes Removed')
                        plt.title('Fraction of nodes remained in different cluster ')  # Size of Largest Connected Component
                        xmax = max([len(x) for x in remove_nodes_list]) / len(G0)
                        plt.xlim(-0.01, 1 + 0.06)
                        plt.legend
                        fig.patch.set_alpha(0)
                        ax.patch.set_alpha(0)
                        plt.savefig(path + 'Cluster_keep_dismantling %s %s %s %s.png' % (graph_type, attack, name,perm), dpi=300)
                        plt.show()
                        # break
                    """骨干结构节点的非关键性"""
                    #度分布+关键节点度分布
                    degdict = dict(G.degree())
                    all_deg_list = list(degdict.values())
                    cluster_deg_list = []
                    for cid in sorted(cluster_dict_new.keys()):
                        cluster = cluster_dict_new[cid]
                        deg_list = [degdict[node] for node in cluster]
                        cluster_deg_list.append(deg_list)
                    degree_list_list = [all_deg_list]+cluster_deg_list
                    name_list =['all node']+["backbone %s"%(i)  if i>0 else 'other nodes' for i in sorted(cluster_dict_new.keys())]
                    colors1 = ['black']+rgba_colors
                    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
                    plt.rcParams['font.family'] = 'Arial'  # 设置字体
                    plt.rcParams['font.size'] = 16  # 设置字号
                    # tab20c = plt.cm.get_cmap('tab20c', 20)
                    dist_list = []
                    for values in degree_list_list:
                        count = Counter(values)
                        dist_list.append(count)
                    all_values = set([y for x in dist_list for y in x])
                    max_value = max(all_values)
                    min_value = min(all_values)
                    max_count = max([max(dist.values(), default=0) for dist in dist_list])
                    fig, ax = plt.subplots(figsize=(6, 6))
                    for dist, color1, name_ in zip(dist_list, colors1, name_list):
                        values = sorted(list(dist.keys()))
                        counts = [dist.get(value, 0) for value in values]
                        ax.plot(values, counts, color=color1, linewidth=1, alpha=0.8)
                        ax.scatter(values, counts, marker='o', s=27, edgecolors='black', alpha=1, linewidth=0.2,
                                   label="%s" % name_, color=color1)
                    # 设置图例
                    legend = ax.legend(frameon=1)
                    frame = legend.get_frame()
                    frame.set_color('none')  # 设置图例边框颜色
                    frame.set_alpha(0)  # 设置图例边框透明
                    # 设置刻度
                    plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
                    plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
                    plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
                    ax.set_xlabel('Degree')
                    ax.set_ylabel('Frequency')
                    ax.set_title('%s Distribution' % "Degree")
                    ax.set_xticks(range(0, max_value + 1, 5))  # 设置x轴刻度，每隔5显示一个刻度
                    ax.set_xscale('log')  # x轴使用对数刻度
                    ax.set_yscale('log')  # y轴使用对数刻度
                    ax.set_xlim([min(0.9, min_value), max_value * 2])
                    ax.set_ylim([0.6, max_count * 2])
                    fig.patch.set_alpha(0)
                    ax.patch.set_alpha(0)
                    plt.savefig(path + 'Noncritical nodes of backbones %s %s %s.png' % (graph_type, attack, name), dpi=300)
                    plt.show()
if __name__=='__main__':
    Silhouette_Analysis(show_case_id = 50)