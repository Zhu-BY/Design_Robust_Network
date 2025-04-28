import pandas as pd
import matplotlib.pyplot as plt
from RL_Algorithm.Environment.dismantlers.dismantlers_with_seeds import dismantle_with_seeds
import networkx as nx
import torch
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import copy
from RL_Algorithm.utils.base_utils import color_get,calculate_metrics,load_csv_net
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['font.family'] = 'Arial'  # 设置字体
plt.rcParams['font.size'] = 16  # 设置字号
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = color_get()
colors.reverse()
custom_colormap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=10)

def draw_dismanting_curve(path,type,dismantling_name,G0, G_list, name_list,r_curve_list=0,x_max0=0):
    # plt.rcParams['font.family'] = 'Arial'  # 设置字体
    # plt.rcParams['font.size'] = 16  # 设置字号
    tab20c = plt.cm.get_cmap('tab20c', 20)
    colors1 = [tab20c(4), tab20c(8), tab20c(12),tab20c(0),tab20c(16)]

    colors1.reverse()
    # colors2.reverse()
    # colors3.reverse()
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
        ax.scatter(x,plot[1], marker='o', s=30, alpha=0.5, c=color1, edgecolors=color1, label='%s: %.2f' % (name,plot[0]))

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
    # plt.savefig(path+" %s %s 2.png"%(type,dismantling_name), dpi=600)
    plt.show()
    return [x[0] for x in r_curve_list],remove_nodes_list # 返回R
def main():
    data_path = '../RL_Algorithm/Design_result/real_network_result/uniform_cost/'
    outpath = './backbone_result/Real_Network/'
    ratio=0.5
    attack ='HDA'
    # names = ['ISP','Germany grid']
    names = ['Germany grid']
    for net_name in names:
        G0,_ = load_csv_net(net_name)
        load_data = torch.load(data_path  + '%s/%s_%s_%s.pth' % (net_name, net_name, ratio, attack))
        if net_name=='ISP':net_name='Sprintlink'
        best_design = [x for x in load_data if x[0] == max([x[0] for x in load_data])][0]
        G1 = best_design[2]
        R_list_now, remove_nodes_list = draw_dismanting_curve(outpath + '%s' % ratio, net_name, attack,G0,[G1],['RL'])
        for G, name, remove_nodes,R_G in zip([G0, G1], ['Original', 'RL'], remove_nodes_list[:],R_list_now):
            """瓦解过程节点特征统计"""
            G_step = copy.deepcopy(G)
            betweenness, degrees, second_order, third_order = calculate_metrics(G_step)
            data = []
            # data.append((betweenness, degrees, second_order, third_order))
            time = 0.2 if net_name == 'Germany grid' else 0.5
            for step in range(int(time*len(G))):
                try:
                    G_step.remove_node(remove_nodes[step])
                except:
                    print("gcc:0")
                gcc = list(max(nx.connected_components(G_step), key=len))
                G_gcc = nx.Graph()
                G_gcc.add_nodes_from(gcc)
                for edge in G_step.edges():
                    if edge[0] in gcc and edge[1] in gcc:
                        G_gcc.add_edge(edge[0], edge[1])

                betweenness, degrees, second_order, third_order = calculate_metrics(G_step)
                data.append((betweenness, degrees, second_order, third_order))
            """介数曲线绘制"""
            df_betweenness = pd.DataFrame([d[0] for d in data])
            df_betweenness.index = range(1, len(df_betweenness) + 1)
            fig, ax = plt.subplots(1, 1, figsize=(7, 2))
            df_betweenness.plot(ax=ax, legend=False, cmap=custom_colormap, alpha=0.9, linewidth=1)
            plt.tight_layout()
            # plt.savefig(outpath + '%s %s %s %s key node metrics.jpg' % (net_name, ratio, attack, name), dpi=300)
            plt.show()
            """对瓦解过程的介数曲线进行聚类"""
            df = df_betweenness.T
            df.fillna(0, inplace=True)
            scaled_data = df.values
            from tslearn.clustering import TimeSeriesKMeans, silhouette_score
            # 将数据转换为 tslearn 能接受的格式
            formatted_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
            for metric in ['euclidean']:
                # 尝试不同的聚类数量
                best_score,best_k,best_model = -1,None,None
                inertias=[]
                for k in range(3,7+1):###############################################################################
                    if name == 'Germany grid': model = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=300,random_state=0) #
                    else:  model = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=300,random_state=2) #
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
                # 按照峰值出现时间由前到后排序
                label_order = sorted(list(cluster_dict.keys()),key=lambda x: list(best_model.cluster_centers_[x].ravel()).index(
                                         max(best_model.cluster_centers_[x].ravel())))
                # time_with_cluster0 = [[list(best_model.cluster_centers_[x].ravel()).index(
                #     max(best_model.cluster_centers_[x].ravel())), cluster_dict[x]] for x in
                #     cluster_dict.keys()]  # 每个cluster的峰值
                # zero_num = len([x for x in time_with_cluster0 if x[0] == 0])  # 峰值为0的数量
                # if zero_num>1: ####
                #     print('zero num >1')
                # new_label = {x: label_order.index(x) for x in label_order}
                # node_cluster_dict = dict()
                # for node, label in zip(df.index, best_model.labels_):
                #     node_cluster_dict[node] = new_label[label]
                # for node in G.nodes():
                #     if node not in node_cluster_dict.keys():
                #         node_cluster_dict[node] = 0
                time_with_cluster0 = [[list(best_model.cluster_centers_[x].ravel()).index(
                    max(best_model.cluster_centers_[x].ravel())), cluster_dict[x]] for x in
                    cluster_dict.keys()]  # 每个cluster的峰值
                zero_num = len([x for x in time_with_cluster0 if x[0] == 0])  # 峰值为0的数量
                time_with_cluster = sorted(time_with_cluster0, key=lambda x: x[0])
                zero_time = []
                for i in range(len(time_with_cluster)):
                    if time_with_cluster[i][0] == 0:
                        zero_time += time_with_cluster[i][1]
                new_time_with_cluster = [[0, zero_time]] + [x for x in time_with_cluster if
                                                            x[0] != 0] if zero_time != [] else time_with_cluster
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

                """可视化聚类结果"""
                # cmap = plt.get_cmap('tab20')
                # rgba_colors = [cmap(2 * 3 + 0), cmap(2 * 0 + 0), cmap(2 * 2 + 0), cmap(2 * 4 + 0),cmap(2 * 5 + 0),cmap(2 * 7 + 0)]  # 红 蓝 绿 紫 灰
                # # if len(cluster_dict) == 3:
                # rgba_colors = rgba_colors[0:(len(cluster_dict)-1)]+[rgba_colors[-1]]
                # rgba_colors.reverse()
                # plt.figure(figsize=(12, 8))
                # for yi,label in zip(range(best_k),label_order):
                #     plt.subplot(best_k, 1, yi + 1)
                #     for xx in formatted_data[best_model.labels_ == label]:
                #         plt.plot(xx.ravel(), "k-", alpha=0.1)
                #     plt.plot(best_model.cluster_centers_[label].ravel(), "r-",color = rgba_colors[yi],linewidth=3,alpha=0.8)
                #     plt.xlim(0, scaled_data.shape[1])
                #     # plt.ylim(-0.01, 0.6)
                #     plt.title(f"Cluster {yi + 1}")
                # plt.suptitle('cluster score: %s'%best_score)
                # plt.tight_layout()
                # plt.savefig(path+'%s %s %s %s betweenness %s cluster.jpg'%(graph_type,id,attack,name,metric),dpi=300)
                # plt.show()
                """绘制瓦解过程不同cluster节点介数整体的变化值"""
                colors = color_get()
                # rgba_colors = [colors[0], colors[1], colors[2], colors[3], colors[6]]  # 红，蓝，棕色，黄，淡紫色
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
                """取每一类节点的最大连通子图"""
                # for cid in cluster_dict.keys():
                #     cluster = cluster_dict[cid]
                #     G_ = copy.deepcopy(G)
                #     backbone_G = G_.subgraph(cluster)
                #     gcc_cluster = list(max(list(nx.connected_components(backbone_G)), key=len))
                #     cluster_dict[cid] = gcc_cluster
                # # 更换标签
                # cluster_dict_new = {new_label[key]: cluster_dict[key] for key in cluster_dict.keys()}
                # # 更换node_cluster_dict标签
                # for node in node_cluster_dict.keys():
                #     if node not in cluster_dict_new[node_cluster_dict[node]]:
                #         node_cluster_dict[node] = 0
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
                for step in range(len(remove_nodes)):
                    G_step.remove_node(remove_nodes[step])
                    bet_now = nx.betweenness_centrality(G_step)
                    gcc = list(max(nx.connected_components(G_step), key=len))
                    if len(gcc) <= 2:
                        break
                    Gcc_list.append(len(gcc) / len(G))
                    for cluster in cluster_dict_new.keys():
                        node_list = cluster_dict_new[cluster]
                        delta_avg_bet = (sum(bet_now[node] for node in node_list if node in G_step.nodes())-sum(bet_last[node] for node in node_list if node in G_step.nodes()))/len(node_list) # 平均介数的变化
                        cluster_bet_delta[cluster].append(delta_avg_bet)
                    bet_last = bet_now
                color = 'black'
                fig, ax1 = plt.subplots(figsize=(6.9, 6))
                ax1.set_xlabel('Fraction of Nodes Removed')
                ax1.set_ylabel('Gcc size')
                # 瓦解曲线
                lines = []
                x = np.linspace(0, len(Gcc_list) - 1, len(Gcc_list))
                x = [i / len(G) for i in x]
                ax1.plot(x, Gcc_list, color=color, linewidth=1, alpha=0.5, zorder=3)
                line= ax1.scatter(x, Gcc_list, marker='o', s=8, alpha=0.5, c=color, edgecolors=color,
                            label='%s: %.2f' % (name, R_G), zorder=3)
                lines.append(line)
                # cluster节点移除比例
                ax2 = ax1.twinx()
                ax2.set_ylabel('Delta of Betweenness')
                for cluster, color in zip(cluster_bet_delta.keys(), rgba_colors):
                    x = np.linspace(0, len(cluster_bet_delta[cluster]) - 1, len(cluster_bet_delta[cluster]))
                    x = [i / len(G0) for i in x]
                    normalized_data_np = cluster_bet_delta[cluster]
                    ax2.plot(x, normalized_data_np, color=color, alpha=0.8, linewidth=1)
                    line=ax2.scatter(x, normalized_data_np, color=color, s=12, alpha=0.99,
                                label='backbone %s' % (cluster) if cluster>0 else 'other nodes')
                    lines.append(line)
                # 设置图例
                labels = [line.get_label() for line in lines]
                if net_name=='Germany grid':
                    # if name=='RL':legend = ax2.legend(lines, labels, loc='best',fontsize=12,frameon=1,bbox_to_anchor=(1, 0.52))
                    legend = ax2.legend(lines, labels, loc='best',fontsize=12,frameon=1,bbox_to_anchor=(1, 1))
                if net_name=='Sprintlink':
                    if name=='RL':
                        legend = ax2.legend(lines, labels, loc='best',fontsize=12,frameon=1,bbox_to_anchor=(0.4, 0.4))
                    else:
                        legend = ax2.legend(lines, labels, loc='best',fontsize=12,frameon=1,bbox_to_anchor=(1, 0.7))
                frame = legend.get_frame()
                frame.set_color('none')  # 设置图例边框颜色
                frame.set_alpha(0)  # 设置图例边框透明
                # 设置y1轴刻度数量
                major_locator = MultipleLocator(base=0.5)  # 主刻度位置间隔
                minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
                ax1.yaxis.set_major_locator(major_locator)
                ax1.yaxis.set_minor_locator(minor_locator)
                # 设置y2轴刻度数量
                # ymax = max([max(l) for l in cluster_bet_delta.values()])
                # ymin = min([min(l) for l in cluster_bet_delta.values()])
                ymax = 0.05 if net_name=='Sprintlink' else 0.08
                ymin = -0.08 if net_name=='Sprintlink' else -0.08
                major_locator = MultipleLocator(base=(ymax - ymin) / 4)  # 主刻度位置间隔
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
                plt.ylim(ymin,ymax)
                # plt.title('Fraction of nodes remained in different cluster ')  # Size of Largest Connected Component
                xmax = 0.3 if net_name=='Germany grid' else 0.5
                plt.xlim(-0.01, xmax + 0.06)
                fig.patch.set_alpha(0)
                ax1.patch.set_alpha(0)
                ax2.patch.set_alpha(0)
                plt.subplots_adjust(right=0.8)
                plt.savefig(outpath + 'Delta_betweenness %s %s %s %s.png' % (net_name, ratio,attack, name), dpi=300)
                plt.show()

if __name__=='__main__':
    main()