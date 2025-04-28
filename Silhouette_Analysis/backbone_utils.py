import copy
import random as rd
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib
import pickle
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

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

def calculate_metrics(G):  # 计算介数、一阶度、二阶度和三阶度
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
        # second_neighbors.difference_update(neighbors + [node])
        second_order_degrees[node] = len(second_neighbors)

        for neighbor in second_neighbors:
            third_neighbors.update(G.neighbors(neighbor))
        # third_neighbors.difference_update(neighbors + list(second_neighbors) + [node])
        third_order_degrees[node] = len(third_neighbors)

    return betweenness, degrees, second_order_degrees, third_order_degrees

def color_get():
    colors = [
        (202 / 255, 58 / 255, 69 / 255),  # 红0
        (67 / 255, 147 / 255, 164 / 255),  # 深蓝1
        (144 / 255, 118 / 255, 115 / 255),  # 棕色2        # (77 / 255, 197 / 255, 109 / 255),  # 绿2
        (253 / 255, 176 / 255, 147 / 255),  # 橘黄3
        (193 / 255, 222 / 255, 156 / 255),  # 浅绿4
        (121 / 255, 167 / 255, 199 / 255),  # 浅蓝5
        (214 / 255, 224 / 255, 237 / 255),  # 淡紫色6
    ]
    return colors


def load_csv_net(name):
    path = "E:\\CSR\\b\Optimal_Graph_Generation\\real_network\\"
    if name in ["weights-dist", 'ISP']:
        path = path + "weights-dist\\" + '1239\\'
        edges = pd.read_csv(path + "weights.intra", sep=' ', header=None)
        # nodes = set(list(edges[0])+list(edges[1]))
        edge_list = [(edges[0][i], edges[1][i]) for i in range(len(edges))]
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        gcc = max(list(nx.connected_components(G)), key=len)
        subgraph = G.subgraph(gcc)
        G = nx.Graph(subgraph)
        G = nx.convert_node_labels_to_integers(G)
        pos = nx.spring_layout(G)
        # import torch
        # pos = torch.load(path+"%s_pos.pth"%name)
        plt.figure()
        nx.draw(G, pos, node_size=1, width=0.05, node_color='r')
        plt.savefig(path + '%s.jpg' % name, dpi=600)
        plt.show()
        return G, pos
    if name in ['Germany grid', 'Germany power']:
        path = path + "Germany grid\\"
        nodes_pos = pd.read_csv(path + "nodes.csv", sep='\t')
        edges = pd.read_csv(path + "edges.csv", sep='\t')
        pos = dict()
        for i in range(len(nodes_pos)):
            pos[nodes_pos['Name'][i]] = np.array([nodes_pos['Lat'][i], nodes_pos['Lon'][i]])
        G = nx.Graph()
        G.add_nodes_from([name for name in list(nodes_pos['Name'])])  ###########增加这句话，然后后面不convert_to_interger
        edge_list = [(edges['From_node'][i], edges['To_node'][i]) for i in range(len(edges))]
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        node_mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
        # 生成新的网络和位置
        H = nx.relabel_nodes(G, node_mapping)
        new_pos = {node_mapping[node]: position for node, position in pos.items()}
        plt.figure()
        nx.draw(H, new_pos, node_size=1, width=0.05, node_color='r')
        plt.savefig(path + '%s.jpg' % name, dpi=600)
        return H, new_pos
    if name in ['Central Chilean power grid', 'Chilean power']:
        path = path + "Central Chilean power grid\\"
        nodes_pos = pd.read_csv(path + "Reduced_node.csv")
        edges = pd.read_csv(path + "Reduced_edge.csv")
        pos = dict()
        for i in range(len(nodes_pos)):
            pos[nodes_pos['Id'][i] - 1] = np.array([nodes_pos['Longitude'][i], nodes_pos['Latitude'][i]])
        G = nx.Graph()
        G.add_nodes_from([Id - 1 for Id in list(nodes_pos['Id'])])  ###########增加这句话，然后后面不convert_to_interger
        edge_list = [(edges['Source'][i] - 1, edges['Target'][i] - 1) for i in range(len(edges))]
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        plt.figure()
        nx.draw(G, pos, node_size=1, width=0.05, node_color='r')
        plt.savefig(path + '%s.jpg' % name, dpi=600)
        return G, pos
    if name in ['USAir97', 'US Air']:
        path = path + "USAir97\\"
        import scipy.io
        M = scipy.io.mmread(path + "inf-USAir97.mtx")
        G = nx.from_scipy_sparse_array(M)
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(G, pos, node_size=1, width=0.05, node_color='r')
        plt.savefig(path + '%s.jpg' % name, dpi=600)
        return G, pos
    if name in ['polblogs', 'Polblogs']:
        path = path + "polblogs\\"
        G = nx.Graph()  # symmetric 表示无向图
        with open(path + "web-polblogs.mtx", 'r') as f:
            for line in f:
                if line.startswith('%'):
                    continue  # 跳过注释行
                parts = line.strip().split()
                if len(parts) == 3:
                    continue  # 跳过维度说明行
                if len(parts) >= 2:
                    u, v = int(parts[0]), int(parts[1])
                    G.add_edge(u - 1, v - 1)  # MatrixMarket 是从 1 开始计数，要转成从 0 开始
        G.remove_edges_from(nx.selfloop_edges(G))
        G = nx.convert_node_labels_to_integers(G)
        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(G, pos, node_size=1, width=0.05, node_color='r')
        plt.savefig(path + '%s.jpg' % name, dpi=600)
        return G, pos
    else:
        nodes_pos = pd.read_csv(path + name + "\\nodes.csv")
        edges = pd.read_csv(path + name + "\\edges.csv")
        pos = dict()
        for i in range(len(nodes_pos)):
            x = nodes_pos[' _pos'][i]
            x = x.strip('array([])')
            x = x.split(",")  # 根据‘，’来将字符串分割成单个元素
            x = list(map(float, x))  # 分离出来的单个元素也是字符串类型的，将其转成浮点
            x = np.array(x)
            # nodes_pos[' _pos'][i] = x
            pos[nodes_pos["# index"][i]] = x
        G = nx.Graph()
        G.add_nodes_from(list(nodes_pos['# index']))  ###########增加这句话，然后后面不convert_to_interger
        edge_list = [(edges['# source'][i], edges[' target'][i]) for i in range(len(edges))]
        G.add_edges_from(edge_list)
        G.remove_edges_from(nx.selfloop_edges(G))
        plt.figure()
        nx.draw(G, pos, node_size=1, width=0.05, node_color='r')
        plt.savefig(path + name + '\\%s.jpg' % name, dpi=600)
        return G, pos


if __name__ == '__main__':
    """检查真实网络load_csv_net"""
    for name in ["ISP", "US Air", "polblogs", 'Central Chilean power grid', 'Deltacom', 'euroroad', 'Germany grid',
                 'Kdl', 'new-york', 'savannah', 'US power', 'washington']:
        G, pos = load_csv_net(name)
        print(name, len(G), len(G.edges()))

    """计算Germany Grid的小世界特性"""
    # G,_= load_csv_net('Germany grid')
    # """判断是否是小世界网络"""
    # # 计算实际网络的聚类系数和平均路径长度
    # actual_clustering = nx.average_clustering(G)
    # actual_path_length = nx.average_shortest_path_length(G)
    # n = G.number_of_nodes()  # 获取节点数
    # m = G.number_of_edges()  # 获取边数
    # # 生成与实际网络相同节点数和边数的随机网络
    # # while True:
    # random_graph = nx.gnm_random_graph(n, m)
    #     # max_gcc = max(list(nx.connected_components(random_graph)),key=len)
    #     # if nx.is_connected(random_graph) or len(max_gcc)>290:
    #     #     break
    # random_clustering = nx.average_clustering(random_graph.subgraph(list(max(nx.connected_components(random_graph),key=len))))
    # random_path_length = nx.average_shortest_path_length(random_graph.subgraph(list(max(nx.connected_components(random_graph),key=len))))
    # # 输出结果
    # print(f"实际网络的聚类系数: {actual_clustering}")
    # print(f"随机网络的聚类系数: {random_clustering}")
    # print(f"实际网络的平均路径长度: {actual_path_length}")
    # print(f"随机网络的平均路径长度: {random_path_length}")
    # # 判断是否为小世界网络
    # if actual_clustering > random_clustering and actual_path_length < random_path_length:
    #     print("该网络可能是小世界网络。")
    # else:
    #     print("该网络可能不是小世界网络。")
    #
    # """ 绘制度分布"""
    # degrees = [G.degree(n) for n in G.nodes()]
    # # 绘制度分布图
    # plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), align='left', rwidth=0.8)
    # plt.title("Degree Distribution")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel("Degree")
    # plt.ylabel("Frequency")
    # plt.show()
    #
    # # 计算度的频率分布
    # degree_counts = np.bincount(degrees)
    # degree_values = np.arange(len(degree_counts))
    # # 过滤掉度为0的情况
    # nonzero_degrees = degree_values[degree_counts > 0]
    # nonzero_counts = degree_counts[degree_counts > 0]
    # # 绘制 log-log 度分布图
    # plt.figure()
    # plt.scatter(np.log(nonzero_degrees), np.log(nonzero_counts), color='blue', marker='o')
    # plt.title("Log-Log Degree Distribution")
    # plt.xlabel("log(Degree)")
    # plt.ylabel("log(Frequency)")
    # plt.show()
    """生成一些WS小世界网络满足聚类系数0.069左右，平均路径9.04左右
    来作为Germany Grid的训练集
    最终选了
    0.6 average path length: 3.515036363636364 ***********************
    0.6 cluster coefficient: 0.06459093795093795**********************

    0.01左右重连比例，多少边有长连接    
    """
    # # G_list = []
    # for k in [2,4]:
    #     print(k)
    #     # for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
    #     for p in [0.01]:
    #         average_path_length_list=[]
    #         clustering_coefficient_list = []
    #         for i in range(1):
    #             # G0 =  generate_network(100,'WS0.3',i)
    #             G0 =  nx.watts_strogatz_graph(100,k=k,p=p)
    #             G_gcc = G0.subgraph(list(max(nx.connected_components(G0),key=len)))
    #             average_path_length_G = nx.average_shortest_path_length(G_gcc)
    #             # 计算平均聚类系数
    #             clustering_coefficient_G = nx.average_clustering(G_gcc)
    #             average_path_length_list.append(average_path_length_G)
    #             clustering_coefficient_list.append(clustering_coefficient_G)
    #             # G_list.append(G0)
    #         print('%s average path length: %s'%(p,sum(average_path_length_list)/len(average_path_length_list)))
    #         print('%s cluster coefficient: %s'%(p,sum(clustering_coefficient_list)/len(average_path_length_list)))
    # print(1)
    #
    # """生成SF2-2.5网络"""
    # import networkx as nx
    # import numpy as np
    # import matplotlib.pyplot as plt
    # def generate_power_law_degree_sequence(gamma, size=1):
    #     """ 生成一个符合幂律分布的度序列 """
    #     return (np.random.zipf(gamma, size=size)+1).clip(max=50)
    # def create_network_from_degree_sequence(deg_seq):
    #     """ 根据给定的度序列创建网络 """
    #     if sum(deg_seq) % 2 != 0:
    #         deg_seq[0] += 1  # 简单调整以确保和为偶数
    #     G = nx.configuration_model(deg_seq)
    #     G = nx.Graph(G)  # 去除自环和平行边
    #     G.remove_edges_from(nx.selfloop_edges(G))
    #     return G
    # def check_scale_law(G):
    #     """ 检查网络的标度律是否在给定范围内 """
    #     degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    #     degree_counts = np.bincount(degree_sequence)
    #     degrees = np.arange(len(degree_counts))
    #     non_zero = degree_counts > 0
    #     degrees = degrees[non_zero]
    #     degree_counts = degree_counts[non_zero]
    #     log_degrees = np.log(degrees)
    #     log_counts = np.log(degree_counts)
    #     try:
    #         slope, intercept = np.polyfit(log_degrees, log_counts, 1)
    #         return -slope
    #     except:
    #         return False
    #
    # try:
    #     with open("SF_graph_list_2.1.pkl", "rb") as f:
    #         G_list, gamma_list=pickle.load(f)
    #         c0 = len(G_list)
    # except:
    #     G_list = []
    #     gamma_list = []
    #     c0=0
    # for i in range(c0,1000):
    #     print(i)
    #     n = 100  # 节点数
    #     target_edges = range(120, 140)  # 目标边数范围
    #     while True:
    #         gamma = np.random.uniform(1.9, 2.2)  # 随机选择一个幂指数
    #         deg_seq = generate_power_law_degree_sequence(gamma, size=n)
    #         G = create_network_from_degree_sequence(deg_seq)
    #         # 检查边数是否合适
    #         true_gamma = check_scale_law(G)
    #         if nx.is_connected(G):
    #             if len(G.edges()) in target_edges:
    #                 if 2.05 <= true_gamma  <= 2.15:
    #                     gamma_list.append(true_gamma)
    #                     G_list.append(G)
    #                     print(i)
    #                     break  # 如果边数合适且标度律也合适，则结束循环
    #     # # 绘制网络
    #     # pos = nx.spring_layout(G)
    #     # nx.draw(G, pos, node_size=20, with_labels=False)
    #     # plt.title(f"Network with {len(G.edges())} edges, Scale Law: {gamma:.2f}")
    #     # plt.show()
    #     with open("SF_graph_list_2.1.pkl", "wb") as f:
    #         pickle.dump([G_list,gamma_list], f)
    #
    # print(1)

    """BA gamma distribution"""
    G_list = []
    gamma_list = []
    for c in range(100):
        G0, _ = generate_network(100, 'SF2.1', c)  # gamma = 1.15-1.77
        G_list.append(G0)
        # plt.figure()
        # nx.draw(G0,_,node_size=5)
        # plt.show()

        degree_sequence = sorted([d for n, d in G0.degree()], reverse=True)
        # 计算度数分布
        degree_counts = np.bincount(degree_sequence)
        degrees = np.arange(len(degree_counts))
        # 筛选非零的度数和频率
        non_zero = degree_counts > 0
        degrees = degrees[non_zero]
        degree_counts = degree_counts[non_zero]
        # 拟合幂律分布
        log_degrees = np.log(degrees)
        log_counts = np.log(degree_counts)
        slope, intercept = np.polyfit(log_degrees, log_counts, 1)
        gamma_list.append(-slope)

        # 绘制度分布的 log-log 拟合线
        plt.figure(figsize=(8, 6))
        plt.scatter(np.log(degrees), np.log(degree_counts), label='Degree Distribution')
        plt.plot(log_degrees, slope * log_degrees + intercept, color='red',
                 label=f'Power Law Fit (slope = {slope:.2f})')
        # 设置图形标题和标签
        plt.title('Log-Log Degree Distribution of BA Network with Power Law Fit')
        plt.xlabel('log(Degree)')
        plt.ylabel('log(Frequency)')
        plt.legend()
        plt.show()

        # 新绘制方法
        import powerlaw

        D_list = []
        for x in np.linspace(1, 26, 26):
            fit = powerlaw.Fit(degree_sequence, xmin=x)
            D_list.append(fit.power_law.D)
        plt.plot(np.linspace(1, 51, 26), D_list, "ro")
        plt.xlabel("$K_{min}$")
        plt.ylabel("$D$")
        plt.yscale("log")
        plt.show()

        fit = powerlaw.Fit(degree_sequence, xmin=2.0)
        kmin = fit.power_law.xmin
        print("kmin:", kmin)
        print("gamma:", fit.power_law.alpha)
        print("D:", fit.power_law.D)
        print('next')

        # import powerlaw
        # degree_sequence = [d for n, d in G0.degree()]
        # fit = powerlaw.Fit(degree_sequence)
        # # 获取幂律拟合参数
        # alpha = fit.alpha  # 幂律指数
        # sigma = fit.sigma  # 拟合标准误差
        # # 打印幂律指数和标准误差
        # print(f'Fitted power law exponent: {alpha}')
        # # print(f'Fitted power law standard deviation: {sigma}')
        # # 可视化对数-对数图
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # fit.power_law.plot_pdf(color='b', linestyle='--', label=f'powerlow Fit(slope={-alpha:.2f}', ax=ax)
        # # # 从 ax 提取数据
        # # line = ax.lines[0]  # 拟合线是第一条线
        # # x_data = line.get_xdata()
        # # y_data = line.get_ydata()
        # # # 确保只处理正值，避免对数问题
        # # valid_indices = (x_data > 0) & (y_data > 0)
        # # x_data = x_data[valid_indices]
        # # y_data = y_data[valid_indices]
        # # # 对 x_data 和 y_data 进行对数变换
        # # log_x_data = np.log(x_data)
        # # log_y_data = np.log(y_data)
        # # # 使用 np.polyfit 进行线性拟合，计算斜率和截距
        # # slope2, intercept2 = np.polyfit(log_x_data, log_y_data, 1)
        # # 绘制节点
        # fit.plot_pdf(color='lightblue', label='Powerlaw data', ax=ax)
        # pdf_x, pdf_y = fit.pdf()
        # # np.polyfit
        # fitted_y = np.exp(intercept) * degrees ** slope/sum(degree_counts[list(degrees).index(min(pdf_x)):])
        # # 绘制 np.polyfit 拟合线
        # plt.plot(degrees, fitted_y, color='r', linestyle='--',label=f'np.polyfit Fit (slope={slope:.2f}')
        # plt.scatter(degrees, np.array(degree_counts)/sum(degree_counts[list(degrees).index(min(pdf_x)):]), label='Original Data',color='orange')
        #
        # plt.xlabel('Degree')
        # plt.ylabel('PDF')
        # plt.legend()
        # plt.savefig('power law.png',dpi=600)
        # plt.show()
        print('next')
    print('finish')

    name = 'Germany grid'
    G, pos = load_csv_net(name)
    pos_new = pos_tran(copy.deepcopy(pos), name)
    import torch

    torch.save(pos_new, 'germany_grid_WGS84.pth')
    plt.figure()
    nx.draw(G, pos_new, node_size=18, node_color='black', edge_color='black', width=0.5)
    plt.savefig('%s topology WGS84.jpg' % (name), dpi=600)
    plt.show()

    import geopandas as gpd
    import matplotlib.pyplot as plt

    germany = gpd.read_file(
        "E:/CSR/b/Optimal_Graph_Generation/real_network/Germany grid/gadm41_DEU_shp/gadm41_DEU_0.shp")
    #
    # fig, ax = plt.subplots(figsize = (8,6))
    # edges = list(G.edges())
    # # 绘制节点
    # for node, (x, y) in pos.items():
    #     ax.plot(x, y, 'o', markersize=3,color='k')  # 'o' 是圆形标记
    #     # ax.text(x, y, node, fontsize=12, ha='center', va='center')
    # # 绘制边
    # for start, end in edges:
    #     start_pos = pos[start]
    #     end_pos = pos[end]
    #     ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-',linewidth=0.8)  # 'k-' 是黑色的线
    # # 设置图的边界
    # # ax.set_xlim(0, 4)
    # # ax.set_ylim(0, 4)
    # # ax.set_aspect('equal')  # 确保x和y轴具有相同的缩放比例，防止图形被拉伸
    # # plt.show()
    fig, ax = plt.subplots(figsize=(8, 6))
    germany.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5, alpha=1)
    germany.plot(ax=ax, color='lightblue', edgecolor='black', linewidth=0.5, alpha=0.2)
    ax.set_title('High-Resolution Map of Germany')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(0.9)  # 设置横向拉长的比例为纵向的一半
    plt.savefig('germany grid with border.jpg', dpi=600)
    plt.show()
    # import geopandas as gpd
    # import matplotlib.pyplot as plt
    # # 加载世界地图数据
    # world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # # 筛选出德国的边境数据
    # germany = world[world.name == "Germany"]
    # # 绘制德国边境
    # # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # germany.plot(ax=ax, color='white', edgecolor='black')
    # ax.set_title('Germany Border')
    # ax.set_xlabel('Longitude')
    # ax.set_ylabel('Latitude')
    # plt.show()

    name = 'washington'
    G, pos = load_csv_net(name)
    pos_new = pos_tran(copy.deepcopy(pos), name)
    import torch

    torch.save(pos_new, 'washington_WGS84.pth')
    plt.figure()
    nx.draw(G, pos_new, node_size=4, node_color='black', edge_color='gray', width=1)
    plt.savefig('%s topology WGS84.jpg' % (name), dpi=600)
    plt.show()

    usa = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_state_500k.zip")
    # 筛选出华盛顿州的边界数据
    washington = usa[usa.NAME == "Washington"]
    # 绘制华盛顿州边界
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    washington.plot(ax=ax, color='white', edgecolor='black')
    ax.set_title('Washington State Border')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.show()