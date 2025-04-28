import dgl
import numpy as np
import torch
import networkx as nx
import warnings
import math
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def graph_batch(graphs,adj):
    # 将图结构数据转换为批量图
    batched_graph = dgl.batch(graphs)
    # 将节点特征提取出来
    node_feats = [g.ndata['feat'] for g in graphs]
    # 转换为张量
    node_feats = torch.stack(node_feats)
    batch_adj = torch.stack(adj)
    return batched_graph, node_feats,batch_adj

def saveargs(args,path):
    content = str(args)
    file_path = path+"/args.txt"   # 将字符串保存到txt文件
    with open(file_path, "w") as file:
        file.write(content)

def virtual_node_g(G):
    g = dgl.from_networkx(G)
    num_nodes = G.number_of_nodes()
    # 添加虚拟节点
    g.add_nodes(1)
    virtual_node_id = num_nodes
    # 添加从节点到虚拟节点的边
    src = torch.arange(num_nodes)
    dst = torch.full((num_nodes,), virtual_node_id)
    g.add_edges(src, dst)
    # 预计算所有节点的基础特征
    degrees_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)
    degrees = np.fromiter(degrees_dict.values(), dtype=np.float32)
    clustering = np.fromiter(clustering_dict.values(), dtype=np.float32)
    # 特征向量中心性与核数提前计算
    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(G)
    except:
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000000)
    core_number_dict = nx.core_number(G)
    eigenvector_feature = np.fromiter(eigenvector_dict.values(), dtype=np.float32)
    core_number_feature = np.fromiter(core_number_dict.values(), dtype=np.float32)
    # 缓存邻居集合，加速计算
    neighbors_cache = {node: set(G.neighbors(node)) for node in G.nodes()}
    # 一维数组单独赋值，效率更高
    degree_feature = degrees
    avg_neighbor_degree_feature = np.zeros(num_nodes, dtype=np.float32)
    avg_neighbor_clustering_feature = np.zeros(num_nodes, dtype=np.float32)
    internal_edges_feature = np.zeros(num_nodes, dtype=np.float32)
    external_edges_feature = np.zeros(num_nodes, dtype=np.float32)
    for node in range(num_nodes):
        neighbors = neighbors_cache[node]
        num_neighbors = len(neighbors)
        neighbor_list = list(neighbors)
        # 邻居平均度与邻居平均聚类系数
        if num_neighbors:
            avg_neighbor_degree_feature[node] = degrees[neighbor_list].mean()
            avg_neighbor_clustering_feature[node] = clustering[neighbor_list].mean()
        else:
            avg_neighbor_degree_feature[node] = 0
            avg_neighbor_clustering_feature[node] = 0
        # Egonet内部边数与外部边数快速计算
        internal_edges = num_neighbors  # 节点到邻居的边数
        for i in range(num_neighbors):
            u = neighbor_list[i]
            neighbors_u = neighbors_cache[u]
            for j in range(i+1, num_neighbors):
                v = neighbor_list[j]
                if v in neighbors_u:
                    internal_edges += 1
        total_degree_egonet = degrees[node] + degrees[neighbor_list].sum()
        external_edges = total_degree_egonet - 2 * internal_edges
        internal_edges_feature[node] = internal_edges
        external_edges_feature[node] = external_edges
    # 高效一次性拼接特征
    input_array = np.stack([
        degree_feature,
        avg_neighbor_degree_feature,
        avg_neighbor_clustering_feature,
        internal_edges_feature,
        external_edges_feature,
        eigenvector_feature,
        core_number_feature
    ], axis=1)
    # 统一归一化特征矩阵
    input_tensor = torch.from_numpy(input_array)
    max_values = input_tensor.max(dim=0).values
    max_values[max_values == 0] = 1
    input_tensor /= max_values
    # 添加虚拟节点特征（全零特征）
    virtual_node_input = torch.zeros((1, 7))
    input_tensor = torch.cat([input_tensor, virtual_node_input], dim=0)
    g.ndata['feat'] = input_tensor
    return g

def action_to_edge(e, n):
    i = int(n - 2 - math.floor(math.sqrt(-8*e + 4*n*(n-1)-7)/2.0 - 0.5))
    temp = (i * (2*n - i - 1)) // 2
    j = int(e + i + 1 - temp)
    return i, j

def edge_mask(G):
    num_nodes = len(G)
    # 获取所有边的节点对，按顺序存储（避免重复检查）
    edges = np.array([sorted((u, v)) for u, v in G.edges()])
    # 创建一个完整的邻接矩阵（上三角部分）
    adj_matrix = np.ones((num_nodes, num_nodes), dtype=bool)
    # 设置对角线为False（自环不算边）
    np.fill_diagonal(adj_matrix, False)
    # 生成上三角索引
    rows, cols = np.triu_indices(num_nodes, k=1)
    # 标记现有的边为False（在对角线以上部分）
    for u, v in edges:
        adj_matrix[u, v] = False
        adj_matrix[v, u] = False  # 因为是无向图，所以需要对称
    # 将矩阵转化为一维数组（行优先）
    matrix = adj_matrix[rows, cols]
    # 转换为torch tensor并返回
    edge_mask_matrix = torch.tensor(matrix, device=device)
    return edge_mask_matrix

def virtual_node_g_cost(G,node_cost,now_cost):
    g = dgl.from_networkx(G)
    num_nodes = G.number_of_nodes()
    # 添加虚拟节点
    g.add_nodes(1)
    virtual_node_id = num_nodes
    # 添加从节点到虚拟节点的边
    src = torch.arange(num_nodes)
    dst = torch.full((num_nodes,), virtual_node_id)
    g.add_edges(src, dst)
    # 预计算所有节点的基础特征
    degrees_dict = dict(G.degree())
    clustering_dict = nx.clustering(G)
    degrees = np.fromiter(degrees_dict.values(), dtype=np.float32)
    clustering = np.fromiter(clustering_dict.values(), dtype=np.float32)
    # 特征向量中心性与核数提前计算
    try:
        eigenvector_dict = nx.eigenvector_centrality_numpy(G)
    except:
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000000)
    core_number_dict = nx.core_number(G)
    eigenvector_feature = np.fromiter(eigenvector_dict.values(), dtype=np.float32)
    core_number_feature = np.fromiter(core_number_dict.values(), dtype=np.float32)
    # 缓存邻居集合，加速计算
    neighbors_cache = {node: set(G.neighbors(node)) for node in G.nodes()}
    # 一维数组单独赋值，效率更高
    degree_feature = degrees
    avg_neighbor_degree_feature = np.zeros(num_nodes, dtype=np.float32)
    norm_node_cost = np.zeros(num_nodes, dtype=np.float32)
    avg_neighbor_clustering_feature = np.zeros(num_nodes, dtype=np.float32)
    internal_edges_feature = np.zeros(num_nodes, dtype=np.float32)
    external_edges_feature = np.zeros(num_nodes, dtype=np.float32)
    for node in range(num_nodes):
        neighbors = neighbors_cache[node]
        num_neighbors = len(neighbors)
        neighbor_list = list(neighbors)
        # 邻居平均度与邻居平均聚类系数
        if num_neighbors:
            avg_neighbor_degree_feature[node] = degrees[neighbor_list].mean()
            avg_neighbor_clustering_feature[node] = clustering[neighbor_list].mean()
        else:
            avg_neighbor_degree_feature[node] = 0
            avg_neighbor_clustering_feature[node] = 0
        # Egonet内部边数与外部边数快速计算
        internal_edges = num_neighbors  # 节点到邻居的边数
        for i in range(num_neighbors):
            u = neighbor_list[i]
            neighbors_u = neighbors_cache[u]
            for j in range(i+1, num_neighbors):
                v = neighbor_list[j]
                if v in neighbors_u:
                    internal_edges += 1
        total_degree_egonet = degrees[node] + degrees[neighbor_list].sum()
        external_edges = total_degree_egonet - 2 * internal_edges
        internal_edges_feature[node] = internal_edges
        external_edges_feature[node] = external_edges
        norm_node_cost[node] = node_cost[node]/now_cost
    # 高效一次性拼接特征
    input_array = np.stack([
        degree_feature,
        avg_neighbor_degree_feature,
        avg_neighbor_clustering_feature,
        internal_edges_feature,
        external_edges_feature,
        eigenvector_feature,
        core_number_feature,
        norm_node_cost
    ], axis=1)
    # 统一归一化特征矩阵
    input_tensor = torch.from_numpy(input_array)
    max_values = input_tensor.max(dim=0).values
    max_values[max_values == 0] = 1
    input_tensor /= max_values
    # 添加虚拟节点特征（全零特征）
    virtual_node_input = torch.zeros((1, 7))
    input_tensor = torch.cat([input_tensor, virtual_node_input], dim=0)
    g.ndata['feat'] = input_tensor
    return g

def edge_mask_cost(G,cost,node_cost):
    edge_cost_list = []
    edges0 = list(G.edges())
    edges=[]
    for edge in edges0:
        if edge[0]>edge[1]:
            edge = (edge[1],edge[0])
        edges.append(edge)
    matrix = []
    for i in range(len(G) - 1):
        for j in range(i + 1, len(G)):
            norm_cost = max(1-(node_cost[i]+node_cost[j])/max(cost,0.00001),-1)
            edge_cost_list.append(norm_cost)
            if (i, j) in edges:
                matrix.append(False)
            elif node_cost[i]+node_cost[j]>cost: # 如果连边成本高于剩余成本，则mask
                matrix.append(False)
            else:
                matrix.append(True)
    edge_mask_matrix = torch.tensor(matrix).to(device)
    edge_cost_tensor = torch.tensor(edge_cost_list).to(device)
    return edge_mask_matrix,edge_cost_tensor.reshape(-1,1).float()

def graph_batch_with_cost(graphs,adj,normalize_cost):
    # 将图结构数据转换为批量图
    batched_graph = dgl.batch(graphs)
    # 将节点特征提取出来
    node_feats = [g.ndata['feat'] for g in graphs]
    # 转换为张量
    node_feats = torch.stack(node_feats)
    batch_adj = torch.stack(adj)
    batch_normalize_length = torch.stack(normalize_cost)
    return batched_graph, node_feats,batch_adj,batch_normalize_length

def virtual_node_g_old(G):
    g = dgl.from_networkx(G)
    # 添加虚拟节点
    g.add_nodes(1)
    num_nodes = len(G)
    virtual_node_id = num_nodes-1+1
    src = torch.tensor([virtual_node_id] * num_nodes)
    dst = torch.arange(0, num_nodes)
    g.add_edges(src, dst)
    input = torch.ones(len(G), 7)  # 最后一项是
    for i in G.nodes():
        input[i, 0] = G.degree()[i]
        input[i, 1] = sum([G.degree()[j] for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)
        input[i, 2] = sum([nx.clustering(G, j) for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)
        egonet = G.subgraph(list(G.neighbors(i)) + [i])
        input[i, 3] = len(egonet.edges())
        input[i, 4] = sum([G.degree()[j] for j in egonet.nodes()]) - 2 * input[i, 3]
    # 特征向量中心性
    try:
        e = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        e = nx.eigenvector_centrality(G, max_iter=1000000)
    # k-core
    k = nx.core_number(G)
    for i in G.nodes():
        input[i, 5] = e[i]
        input[i, 6] = k[i]
    # 归一化
    for i in range(len(input[0])):
        if max(input[:, i]) != 0:  #########  max or sum
            input[:, i] = input[:, i] / max(input[:, i])  # 归一化
    ndata = input
    #虚拟节点特征
    virtual_node_input = torch.zeros(1,7)
    input = torch.concat((input,virtual_node_input),dim=0)
    g.ndata['feat'] = input
    return g


# def virtual_node_g_old(G):
#     g = dgl.from_networkx(G)
#     num_nodes = G.number_of_nodes()
#     # 添加虚拟节点
#     g.add_nodes(1)
#     virtual_node_id = num_nodes
#     # 添加从节点到虚拟节点的边
#     dst = torch.arange(num_nodes)
#     src = torch.full((num_nodes,), virtual_node_id)
#     g.add_edges(src, dst)
#     # 预计算所有节点的基础特征
#     degrees_dict = dict(G.degree())
#     clustering_dict = nx.clustering(G)
#     degrees = np.fromiter(degrees_dict.values(), dtype=np.float32)
#     clustering = np.fromiter(clustering_dict.values(), dtype=np.float32)
#     # 特征向量中心性与核数提前计算
#     try:
#         eigenvector_dict = nx.eigenvector_centrality_numpy(G)
#     except:
#         eigenvector_dict = nx.eigenvector_centrality(G, max_iter=1000000)
#     core_number_dict = nx.core_number(G)
#     eigenvector_feature = np.fromiter(eigenvector_dict.values(), dtype=np.float32)
#     core_number_feature = np.fromiter(core_number_dict.values(), dtype=np.float32)
#     # 缓存邻居集合，加速计算
#     neighbors_cache = {node: set(G.neighbors(node)) for node in G.nodes()}
#     # 一维数组单独赋值，效率更高
#     degree_feature = degrees
#     avg_neighbor_degree_feature = np.zeros(num_nodes, dtype=np.float32)
#     avg_neighbor_clustering_feature = np.zeros(num_nodes, dtype=np.float32)
#     internal_edges_feature = np.zeros(num_nodes, dtype=np.float32)
#     external_edges_feature = np.zeros(num_nodes, dtype=np.float32)
#     for node in range(num_nodes):
#         neighbors = neighbors_cache[node]
#         num_neighbors = len(neighbors)
#         neighbor_list = list(neighbors)
#         # 邻居平均度与邻居平均聚类系数
#         if num_neighbors:
#             avg_neighbor_degree_feature[node] = degrees[neighbor_list].mean()
#             avg_neighbor_clustering_feature[node] = clustering[neighbor_list].mean()
#         else:
#             avg_neighbor_degree_feature[node] = 0
#             avg_neighbor_clustering_feature[node] = 0
#         # Egonet内部边数与外部边数快速计算
#         internal_edges = num_neighbors  # 节点到邻居的边数
#         for i in range(num_neighbors):
#             u = neighbor_list[i]
#             neighbors_u = neighbors_cache[u]
#             for j in range(i+1, num_neighbors):
#                 v = neighbor_list[j]
#                 if v in neighbors_u:
#                     internal_edges += 1
#         total_degree_egonet = degrees[node] + degrees[neighbor_list].sum()
#         external_edges = total_degree_egonet - 2 * internal_edges
#         internal_edges_feature[node] = internal_edges
#         external_edges_feature[node] = external_edges
#     # 高效一次性拼接特征
#     input_array = np.stack([
#         degree_feature,
#         avg_neighbor_degree_feature,
#         avg_neighbor_clustering_feature,
#         internal_edges_feature,
#         external_edges_feature,
#         eigenvector_feature,
#         core_number_feature
#     ], axis=1)
#     # 统一归一化特征矩阵
#     input_tensor = torch.from_numpy(input_array)
#     max_values = input_tensor.max(dim=0).values
#     max_values[max_values == 0] = 1
#     input_tensor /= max_values
#     # 添加虚拟节点特征（全零特征）
#     virtual_node_input = torch.zeros((1, 7))
#     input_tensor = torch.cat([input_tensor, virtual_node_input], dim=0)
#     g.ndata['feat'] = input_tensor
#     return g

def create_edge_indices(n):
    edges = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            edges.append((i, j))
    edges = torch.tensor(edges, dtype=torch.long).to(device)
    return edges

if __name__=='__main__':
    print('This file is rl_utils')