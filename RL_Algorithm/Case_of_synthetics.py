import argparse
import networkx as nx
import torch
import numpy as np
from RL_Algorithm.Environment.dismantlers.dismantlers_ import dismantle
from RL_Algorithm.model import PPO
import matplotlib.pyplot as plt
import copy
from collections import Counter
from itertools import count
import pickle
from RL_Algorithm.Environment.envs import env
from RL_Algorithm.utils.base_utils import color_get
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
plt.rcParams['font.family'] = 'Arial'  # 设置字体
plt.rcParams['font.size'] = 16  # 设置字号

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def draw_dismanting_curve(path, type, dismantling_name, G0, name_list, r_curve_list, x_max0=0):
    # from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    colors = color_get()
    colors1 = [colors[6], colors[1], colors[0]]
    N = len(G0)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for plot, color1 in zip(r_curve_list, colors1):
        x = np.linspace(0, len(plot[1]) - 1, len(plot[1]))
        x = [i / N for i in x]
        ax.plot(x, plot[1], color=color1, linewidth=1, alpha=0.8)
    markers = ['o', 'd', 'd']
    ss = [50, 70, 70]
    for plot, name, color1, marker, s in zip(r_curve_list, name_list, colors1, markers, ss):
        x = np.linspace(0, len(plot[1]) - 1, len(plot[1]))
        x = [i / N for i in x]
        ax.scatter(x, plot[1], marker=marker, s=s, facecolors='none', edgecolors=color1, alpha=1, linewidth=1.1,
                   label='%s: %.2f' % (name, plot[0]))
    legend = ax.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('none')
    frame.set_alpha(0)
    major_locator = MultipleLocator(base=0.5)
    minor_locator = AutoMinorLocator(n=2)
    plt.gca().yaxis.set_major_locator(major_locator)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    major_locator = MultipleLocator(base=0.05)
    minor_locator = AutoMinorLocator(n=2)
    plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().tick_params(which='both', direction='in', top=True, right=True)
    plt.tick_params(axis='both', which='major', length=6, width=1.5)
    plt.tick_params(axis='both', which='minor', length=3, width=1.5)
    plt.title('Network Disintegration Curve for %s' % dismantling_name)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Residual robustness')  # Size of Largest Connected Component
    xmax = max([len(x[1]) for x in r_curve_list]) / len(G0) if x_max0 == 0 else x_max0
    plt.xlim(-0.002, xmax + 0.01)
    plt.ylim(0, 1.05)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig(path + "Dismantling Curve %s %s.png" % (type, dismantling_name), dpi=600)
    plt.show()
    return [x[0] for x in r_curve_list]  # 返回R值

def draw_pyvis_graph_fix_pos(path,G0,G1,name,G2=0):
    from pyvis.network import Network
    edge_adds1 = [edge for edge in G1.edges if edge not in G0.edges]
    edge_adds_nodes1 = [x[0] for x in edge_adds1] + [x[1] for x in edge_adds1]
    if G2!=0:
        edge_adds2 = [edge for edge in G2.edges if edge not in G1.edges]
        edge_adds_nodes2 = [x[0] for x in edge_adds2] + [x[1] for x in edge_adds2]

    rgba_colors = color_get()
    hex_colors = ['#%02x%02x%02x' % (int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255)) for
                  rgba_color in rgba_colors]

    other_color=hex_colors[-1]
    add_color1 = hex_colors[1]
    add_color2 = hex_colors[0]
    width0 = 8
    width1 = 8 if G2!=0 else 16
    width2 = 16

    net = Network()
    net.toggle_physics(False)
    # 添加点
    G1_nodes = list(G1.nodes())
    degrees = [G1.degree(n) for n in G1_nodes]
    sizes = [(d+1) * 4 for d in degrees]
    for node, size in zip(list(G1.nodes()), sizes):
        node_color='gray'
        if node in edge_adds_nodes1:
            node_color=add_color1
        if G2!=0:
            if node in edge_adds_nodes2:
                node_color = add_color2
        net.add_node(node, color=node_color, size=size)
    # 添加边
    for edge in G0.edges():
        source, target = edge
        net.add_edge(int(source), int(target), width=width0, color=other_color)
    for edge in edge_adds1:
        source, target = edge
        net.add_edge(int(source), int(target), width=width1, color=add_color1)
    if G2!=0:
        for edge in edge_adds2:
            source, target = edge
            net.add_edge(int(source), int(target), width=width2, color=add_color2)
    # 设置节点位置
    import json
    if name=='Step 0' or name=='Step 50':
        with open('./Design_result/'+'one_case_position.json', 'r') as f:
            positions = json.load(f)
    else:
        with open("../Silhouette_Analysis/backbone_result/N600/"+'position_shown_backbone.json', 'r') as f:
            positions = json.load(f)
    for node_id, pos in positions.items():
        net.get_node(int(node_id))['x'] = pos['x']
        net.get_node(int(node_id))['y'] = pos['y']
    net.show_buttons(filter_=['physics'])
    net.save_graph(path+'%s topology'%(name)+".html")
    print('finish')

def draw_curve(path,name,label_list,y_list,x_list,y_label,x_label,ind=0):
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    plt.rcParams['font.family'] = 'Arial'  # 设置字体
    plt.rcParams['font.size'] = 16  # 设置字号
    # tab20c = plt.cm.get_cmap('tab20c', 20)
    # colors1 = [tab20c(0), tab20c(4), tab20c(8), tab20c(12),tab20c(16)]
    # colors3 = [tab20c(1), tab20c(5), tab20c(9), tab20c(13) ,tab20c(17)]
    # colors2 = [tab20c(3), tab20c(7), tab20c(11), tab20c(15),tab20c(18)]
    colors1 = plt.cm.get_cmap('tab10')
    colors1 = [colors1(i) for i in [0,1,2,4,5,6,7]]
    if ind!=0:
        colors1 = [colors1[0],colors1[ind]]

    # if len(y_list)>1:
    #     colors1.reverse()
        # colors2.reverse()
        # colors3.reverse()
    alpha = 0.5
    markers = ['o','s','d','v','X','^','|']
    fig, ax = plt.subplots(figsize=(10, 6))
    # fig, ax = plt.subplots()
    for x,y,label,color1 in zip(x_list,y_list,label_list,colors1):
        ax.plot(x,y, color=color1, linewidth=1,alpha=alpha)
    for x,y,label,color1,marker in zip(x_list,y_list,label_list,colors1,markers):
        if label_list[0] != 0:
            ax.scatter(x,y, marker=marker, s=30, alpha=alpha, c=color1, edgecolors=color1, label=label)
        else:
            ax.scatter(x, y, marker=marker, s=30, alpha=alpha, c=color1, edgecolors=color1)

    legend = ax.legend(frameon=1)
    frame = legend.get_frame()
    frame.set_color('none')  # 设置图例边框颜色
    # frame.set_edgecolor('none')  # 设置图例边缘颜色
    frame.set_alpha(0)  # 设置图例边框透明
    # legend = ax.legend(frameon=1)
    # major_locator = MultipleLocator(base=0.5)  # 主刻度位置间隔
    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
    # plt.gca().yaxis.set_major_locator(major_locator)
    plt.gca().yaxis.set_minor_locator(minor_locator)
    # major_locator = MultipleLocator(base=0.2)  # 主刻度位置间隔
    minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
    # plt.gca().xaxis.set_major_locator(major_locator)
    plt.gca().xaxis.set_minor_locator(minor_locator)
    plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
    plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
    plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
    plt.title('%s'%y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label) #Size of Largest Connected Component
    if label_list[0]!=0:
        plt.legend(fontsize=8, loc='best')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    plt.savefig(path+"%s.jpg"%(name), dpi=600)
    plt.show()
    return 0
def main():
    path = "./Design_result/synthetic_result/"
    graph_types = ['BA']
    attack_types =['HDA']
    n=600
    for graph_type in graph_types:
        for attack in attack_types:
            data_loaded = torch.load('./Design_result/synthetic_result/N600/%s_%s_%s_rl_result.pth'%(attack,graph_type,n))
            print(attack,graph_type)
            G_list = data_loaded[0][-1]
            dismantle_number= 1 if attack not in ['GND', 'GNDR'] else 4
            """设计过程鲁棒性曲线图"""
            R_list_curve_list = [dismantle(attack,G,dismantle_number) for G in G_list] # 设计过程鲁棒性变化
            R_list =[x[0] for x in  R_list_curve_list]
            """设计过程三个网络图及其度分布图"""
            G_show_list = [G_list[0],G_list[50],G_list[-1]]
            draw_pyvis_graph_fix_pos(path+"%s %s %s %s " % (n,0, graph_type,attack),G_show_list[0],G_show_list[0],'Step 0')
            draw_pyvis_graph_fix_pos(path+"%s %s %s %s " % (n,0, graph_type,attack),G_show_list[0], G_show_list[1], 'Step 50')
            draw_pyvis_graph_fix_pos(path+"%s %s %s %s " % (n,0, graph_type,attack),G_show_list[0], G_show_list[1], 'Step 101',G_show_list[2])
            degree_list_list = [list(dict(G.degree()).values()) for G in G_show_list]
            """绘制三个网络图度分布曲线在一起"""
            name_list = ['Initial', 'Step 50', 'Step 101']
            colors = color_get()
            colors1 = [colors[2], colors[1], colors[0]]
            dist_list = []
            for values in degree_list_list:
                count = Counter(values)
                dist_list.append(count)
            all_values = set([y for x in dist_list for y in x])
            max_value = max(all_values)
            min_value = min(all_values)
            max_count = max([max(dist.values(), default=0) for dist in dist_list])
            fig, ax = plt.subplots(figsize=(6, 6))
            markers = ['o','+','d']
            for dist, color1,name,z,marker in zip(dist_list, colors1,name_list,[1,2,3],markers):
                values = sorted(list(dist.keys()))
                counts = [dist.get(value, 0) for value in values]
                # ax.plot(values, counts, color=color1, linewidth=1, alpha=0.5,zorder=z)
                # ax.scatter(values, counts, marker='o', s=150, edgecolors='black', alpha=1, linewidth=0.2, label="%s" % name, color=color1,zorder=z)
                if marker!='+':
                    ax.scatter(values, counts, marker=marker, s=150, edgecolors=color1, alpha=0.8, linewidth=1, label="%s" % name, facecolors='none',zorder=z)
                else:
                    ax.scatter(values, counts, marker=marker, s=150, edgecolors='none', alpha=0.8, linewidth=1,  label="%s" % name, color=color1, zorder=z)
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
            ax.set_xlim([min(0.7, min_value), max_value * 2])
            ax.set_ylim([0.6, max_count * 2])
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            plt.savefig(path+'%s %s %s %s degree distribution xylog.png' % (n,0, graph_type,attack),dpi=600)
            plt.show()

            """绘制三个网络的瓦解曲线"""
            name_list = ['Initial','RL-step 50','RL-step 101']
            # 绘制5个网络的瓦解曲线
            draw_dismanting_curve(path,'%s %s %s'%(n,0 ,graph_type) , attack,
                                  G_show_list[0], name_list, [R_list_curve_list[0],R_list_curve_list[50],R_list_curve_list[100]],0.1)

            """绘制鲁棒性与效率关系图"""
            Eff_list = []
            for G in G_list:
                Eff = nx.global_efficiency(G);Eff_list.append(Eff)
            y = R_list
            x = Eff_list
            z = list(range(0,len(R_list)))
            from matplotlib.ticker import MultipleLocator, AutoMinorLocator
            plt.rcParams['font.family'] = 'Arial'  # 设置字体
            plt.rcParams['font.size'] = 16  # 设置字号
            # fig, ax = plt.subplots(figsize=(8.5, 6))
            fig, ax = plt.subplots(figsize=(9.36, 4.8))
            colors=color_get()
            from matplotlib.colors import LinearSegmentedColormap
            custom_colormap = LinearSegmentedColormap.from_list("custom_cmap", [colors[-1],colors[5],colors[1],colors[0]], N=10)
            norm = plt.Normalize(vmin=min(z), vmax=max(z))
            colors = custom_colormap(norm(z))
            scatter = ax.scatter(x, y, marker='o', s=80, alpha=1,  facecolors='none',linewidth=1,edgecolors=colors)
            plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_colormap), label='step') # 添加颜色条
            # 设置图例
            # legend = ax.legend(frameon=1)
            frame = legend.get_frame()
            frame.set_color('none')  # 设置图例边框颜色
            frame.set_alpha(0)  # 设置图例边框透明
            # 设置y轴刻度数量
            major_locator = MultipleLocator(base=20)  # 主刻度位置间隔
            minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
            plt.gca().yaxis.set_major_locator(major_locator)
            plt.gca().yaxis.set_minor_locator(minor_locator)
            # 设置x轴刻度数量
            major_locator = MultipleLocator(base=0.01)  # 主刻度位置间隔
            minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
            plt.gca().xaxis.set_major_locator(major_locator)
            plt.gca().xaxis.set_minor_locator(minor_locator)
            # 设置刻度样式
            plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
            plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
            plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
            plt.xlabel('Efficiency')
            plt.ylabel('Robustness') #Size of Largest Connected Component
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            plt.savefig(path+"Robustness vs efficiency %s %s %s %s.jpg" % (n,0, graph_type,attack), dpi=600)
            plt.show()
            """添加的连边的节点距离与网络半径的关系"""
            edge_list = []
            distance_list = []
            diameter_list = []
            for i in range(len(G_list)-1):
                G0 = G_list[i]
                G1 = G_list[i+1]
                for edge in G1.edges():
                    if edge not in G0.edges():
                        edge_list.append(edge)
                        break
                u = edge[0]
                v = edge[1]
                distance = nx.shortest_path_length(G0,u,v)
                diameter = nx.diameter(G0)
                distance_list.append(distance)
                diameter_list.append(diameter)
            y_list = [diameter_list,distance_list]
            y_name = ['Diameter','s-t distance']
            x_list = [list(range(1,len(R_list))),list(range(1,len(R_list))),list(range(1,len(R_list)))]
            colors = color_get()
            colors1 = [colors[1],colors[0],colors[6],colors[2],colors[6],colors[3]]
            from matplotlib.ticker import MultipleLocator, AutoMinorLocator
            plt.rcParams['font.family'] = 'Arial'  # 设置字体
            plt.rcParams['font.size'] = 16  # 设置字号
            fig, ax = plt.subplots(figsize=(6, 6))
            for x,plot, color1 in zip(x_list,y_list, colors1):
                ax.plot(x, plot, color=color1, linewidth=1, alpha=0.8)
            for x,plot, label, color1 in zip(x_list,y_list, y_name, colors1):
                ax.scatter(x, plot, marker='o', s=20,  c=color1, edgecolors='black',alpha=1,linewidth=0.2,
                           label=label)
                # 设置图例
            legend = ax.legend(frameon=1,fontsize=12,loc='best')
            frame = legend.get_frame()
            frame.set_color('none')  # 设置图例边框颜色
            frame.set_alpha(0)  # 设置图例边框透明
            # 设置x轴刻度数量
            major_locator = MultipleLocator(base=40)
            minor_locator = AutoMinorLocator(n=2)  # 次刻度数量
            plt.gca().xaxis.set_major_locator(major_locator)
            plt.gca().xaxis.set_minor_locator(minor_locator)
            # 设置刻度样式
            plt.gca().tick_params(which='both', direction='in', top=True, right=True)  # 刻度在内
            plt.tick_params(axis='both', which='major', length=6, width=1.5)  # 主刻度线
            plt.tick_params(axis='both', which='minor', length=3, width=1.5)  # 次刻度线
            # 设置标签与标题
            plt.xlabel('Step')
            fig.patch.set_alpha(0)
            ax.patch.set_alpha(0)
            plt.savefig(path+"network metrics %s %s %s %s.jpg" % (n,0, graph_type,attack), dpi=600)
            plt.show()

if __name__=='__main__':
    main()