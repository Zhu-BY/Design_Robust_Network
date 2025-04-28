import copy
import igraph as ig
import heapq
import numpy as np
from collections import Counter

def HDA(G0,p_n=1):
    G=copy.deepcopy(G0)
    # igraph
    g = ig.Graph(directed=False)
    g.add_vertices(list(G.nodes))
    g.add_edges(list(G.edges))
    # 找到最大团
    gcc0 = len(max(g.connected_components(), key=len))
    gcc = gcc0
    r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
    # removed_nodes = []  # 如果需要记录removed_nodes则需要对G同时做瓦解
    qc=0 # qc的值
    while(gcc>p_n):
        # igraph
        deg = g.degree()
        max_deg = max(deg)
        indices_of_2 = [index for index, element in enumerate(deg) if element == max_deg]
        # node_with_deg_max_g = rd.choice(indices_of_2)
        node_with_deg_max_g = indices_of_2[0]
        # node_with_deg_max =list(G.nodes())[node_with_deg_max_g]
        # G.remove_node(node_with_deg_max)
        # removed_nodes.append(node_with_deg_max)
        g.delete_vertices(node_with_deg_max_g)
        gcc = len(max(g.connected_components(), key=len))
        qc+=1
        # gcc = len(list(max(nx.connected_components(G), key=len))) # 当前最大连通子团
        r_list.append(gcc/len(G0))
    # 计算 R 值
    R_Deg = sum(r_list)
    return R_Deg,r_list

def HBA(G0, p_n=1,approx_cutoff=None,batch_frac=0.0):
    """Robustness R under highest‑betweenness attack, NetworkX → igraph."""
    # --- 转换 ---
    nodes   = list(G0.nodes())
    idx_map = {v: i for i, v in enumerate(nodes)}
    edges   = [(idx_map[u], idx_map[v]) for u, v in G0.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    N = g.vcount()

    giant  = g.components().giant().vcount()
    r_hist = [giant / N]

    while giant > p_n and g.vcount():
        # 1) 介数（可选 cutoff≈logN 做近似）
        if approx_cutoff is None:
            bt = np.asarray(g.betweenness(directed=False))
        else:
            bt = np.asarray(g.betweenness(directed=False, cutoff=approx_cutoff))

        # 2) 删除 batch_frac·N 个最高介数节点（默认 1 个）
        k = max(1, int(batch_frac * g.vcount())) if batch_frac > 0 else 1
        g.delete_vertices(bt.argsort()[::-1][:k].tolist())

        # 3) 更新
        giant = g.components().giant().vcount() if g.vcount() else 0
        r_hist.append(giant / N)

    Rc = float(np.sum(r_hist))
    return Rc, r_hist

def compute_CI_1node(adj, degrees, valid, node, l=2):
    """
    根据邻接表 adj、度数列表 degrees、valid(是否被移除)来计算单个节点node的CI值。
    l=2 时: CI(i) = (d(i)-1)* sum_{j in Ball2(i)} (d(j)-1).
    Ball2(i) 大体是 "i邻居的邻居 - i的邻居 - i本身"。
    """
    if not valid[node]:
        return -1  # 已被移除

    d_i = degrees[node]
    if d_i <= 0:
        return 0  # 如果这个地方设置为0，那么就和之前的算法一样了，可能会移除孤立节点.如果设置为-1就会忽略孤立节点####*************##############

    # 1阶邻居
    neighbors = adj[node]

    # 2阶邻居: 邻居的邻居，去除1阶邻居和自身
    ball_2 = set()
    for nb in neighbors:
        if valid[nb]:
            for nb2 in adj[nb]:
                if valid[nb2] and nb2 not in neighbors and nb2 != node:
                    ball_2.add(nb2)

    # 计算 CI
    return (d_i - 1) * sum((degrees[j] - 1) for j in ball_2)

def largest_cc_size(adj, valid):
    """
    计算当前剩余节点(由 valid 标识)的最大连通分量大小。
    使用 BFS/DFS 即可。
    """
    visited = set()
    max_size = 0
    for start in range(len(adj)):
        if not valid[start]:
            continue
        if start not in visited:
            # BFS
            queue = [start]
            visited.add(start)
            comp_size = 1
            idx = 0
            while idx < len(queue):
                cur = queue[idx]
                idx += 1
                for nb in adj[cur]:
                    if valid[nb] and nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
                        comp_size += 1
            max_size = max(max_size, comp_size)
    return max_size

def CI2(G0, p_n=1, l=2):
    Gnx = copy.deepcopy(G0)
    G = ig.Graph(directed=False)
    G.add_vertices(list(Gnx.nodes))
    G.add_edges(list(Gnx.edges))
    """
    优化后的CI拆解算法：
    - 使用最大堆(优先队列)存储CI值
    - 每次移除CI最大的节点后，只局部更新受影响节点的CI
    - 重新计算GCC时，用 BFS/DFS 对剩余节点做一次即可
      (如需更高级的动态连通分量维护，可自行实现)
    """
    # 拷贝图，转为邻接表
    # G = copy.deepcopy(G0)
    n = G.vcount()
    edges = G.get_edgelist()

    # 构建邻接表
    adj = [set() for _ in range(n)]
    for e in edges:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])

    # 有效性标记，表示节点是否还在图中
    valid = [True] * n
    # 度数
    degrees = [G.degree(i) for i in range(n)]

    # 初始最大连通分量
    gcc0 = len(max(G.connected_components(), key=len))
    r_list = [gcc0 / n]

    # 最大堆(在python里用负值来实现)
    heap = []
    for i in range(n):
        ci_val = compute_CI_1node(adj, degrees, valid, i, l=l)
        heap.append((-ci_val, i))
    heapq.heapify(heap)

    gcc = gcc0
    removed_count = 0

    # 不断移除CI最大的节点，直到GCC <= p_n
    while gcc > p_n:
        # 弹出堆顶(可能是过期的)
        ci_val, node = heapq.heappop(heap)
        ci_val = -ci_val

        # 判断该节点是否有效、CI是否过期
        if not valid[node]:
            # 已经被移除过，跳过
            continue
        current_ci = compute_CI_1node(adj, degrees, valid, node, l=l)
        if abs(current_ci - ci_val) > 1e-9:
            # 过期，需要更新后重新选堆顶
            # 这里直接放回新的值即可
            new_ci = compute_CI_1node(adj, degrees, valid, node, l=l)
            heapq.heappush(heap, (-new_ci, node))
            continue

        # 真正移除该节点
        valid[node] = False
        removed_count += 1

        # 更新邻居的度，并对其CI值进行更新
        for nb in adj[node]:
            if valid[nb]:
                degrees[nb] -= 1
                # 只要度数有变化，CI可能变化，需要放入堆做更新
                new_ci = compute_CI_1node(adj, degrees, valid, nb, l=l)
                heapq.heappush(heap, (-new_ci, nb))

        # 清空这个节点的邻接
        for nb in adj[node]:
            adj[nb].discard(node)
        adj[node].clear()

        # 计算新的最大连通分量
        gcc = largest_cc_size(adj, valid)
        r_list.append(gcc / n)

        if gcc <= p_n:
            break

    Rc_CI = sum(r_list)
    return Rc_CI, r_list

# def HBA(G0,p_n=1):
#     G=copy.deepcopy(G0)
#     g = ig.Graph(directed=False)
#     g.add_vertices(list(G.nodes))
#     try:
#         g.add_edges(list(G.edges))
#     except:
#         print(1)
#     # 找到最大团
#     gcc0 = len(max(g.connected_components(), key=len))
#     gcc = gcc0
#     r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
#     # removed_nodes = []
#     qc=0 # qc的值
#     while(gcc>p_n):
#         # igraph
#         ig_betweenness = g.betweenness(directed=False)
#         c= Counter(ig_betweenness)
#         if c[max(ig_betweenness)]>1:
#             node_with_bet_max_gs = [ind for ind in range(len(ig_betweenness)) if ig_betweenness[ind] ==max(ig_betweenness)]
#             # rd.seed()
#             # node_with_bet_max_g = rd.choice(node_with_bet_max_gs)
#             node_with_bet_max_g = node_with_bet_max_gs[0]
#         else:
#             node_with_bet_max_g = ig_betweenness.index(max(ig_betweenness))
#         # node_with_bet_max_G = list(G.nodes())[node_with_bet_max_g]
#         # G.remove_node(node_with_bet_max_G)
#         # removed_nodes.append(node_with_bet_max_G)
#         g.delete_vertices(node_with_bet_max_g)
#         qc+=1
#         gcc = len(max(g.connected_components(), key=len)) # 当前最大连通子团
#         r_list.append(gcc/len(G0))
#     # 计算 R 值
#     Rc_bet = sum(r_list)
#     return Rc_bet,r_list

# def ig_CI2(G0,p_n=1,l=2):  # 可能会移除本身没有边的孤立节点节点
#     G=copy.deepcopy(G0)
#     # igraph
#     g = ig.Graph(directed=False)
#     g.add_vertices(list(G.nodes))
#     g.add_edges(list(G.edges))
#     # 找到最大团
#     gcc0 = len(max(g.connected_components(), key=len))
#     gcc = gcc0
#     r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
#     # removed_nodes = []  # 如果需要记录removed_nodes则需要对G同时做瓦解
#     qc=0 # qc的值
#     node_list = []
#     while(gcc>p_n):
#         # igraph
#         CI = dict() # 计算当前网络的CI值
#         Nl_1 = g.neighborhood(g.vs.indices,order=l-1)
#         Nl = g.neighborhood(g.vs.indices,order=l)
#         Ball_l = [[x for x in Nl[i] if x not in Nl_1[i]] for i in range(len(g.vs.indices))]
#         for i in g.vs.indices:
#             CI[i] = (g.degree()[i]-1)*sum([g.degree()[j]-1 for j in Ball_l[i]])
#         max_CI = max(list(CI.values()))
#         if max_CI==0:
#             print(1)
#         u_list = [k for k in list(CI.keys()) if CI[k] == max_CI]
#         # node_with_CI_max_g = rd.choice(u_list)
#         node_with_CI_max_g = u_list[0]
#         node_list.append(node_with_CI_max_g)
#         # node_with_CI_max_G = list(G.nodes())[node_with_CI_max_g]
#         # G.remove_node(node_with_CI_max_G)
#         # removed_nodes.append(node_with_CI_max_G)
#         qc+=1
#         g.delete_vertices(node_with_CI_max_g)
#         gcc = len(max(g.connected_components(), key=len))
#         r_list.append(gcc/len(G0))
#     # 计算 R 值
#     Rc_CI = sum(r_list)
#     return Rc_CI,r_list,node_list

def HDA_with_seeds(G0,p_n=1):
    G=copy.deepcopy(G0)
    # igraph
    g = ig.Graph(directed=False)
    g.add_vertices(list(G.nodes))
    g.add_edges(list(G.edges))
    # 找到最大团
    gcc0 = len(max(g.connected_components(), key=len))
    gcc = gcc0
    r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
    removed_nodes = []  # 如果需要记录removed_nodes则需要对G同时做瓦解
    qc=0 # qc的值
    while(gcc>p_n):
        # igraph
        deg = g.degree()
        max_deg = max(deg)
        indices_of_2 = [index for index, element in enumerate(deg) if element == max_deg]
        # node_with_deg_max_g = rd.choice(indices_of_2)
        node_with_deg_max_g = indices_of_2[0]
        node_with_deg_max =list(G.nodes())[node_with_deg_max_g]
        G.remove_node(node_with_deg_max)
        removed_nodes.append(node_with_deg_max)
        g.delete_vertices(node_with_deg_max_g)
        gcc = len(max(g.connected_components(), key=len))
        qc+=1
        # gcc = len(list(max(nx.connected_components(G), key=len))) # 当前最大连通子团
        r_list.append(gcc/len(G0))
    # 计算 R 值
    R_Deg = sum(r_list)
    return R_Deg,r_list,removed_nodes

def CI2_with_seeds(G0,p_n=1,l=2):
    G=copy.deepcopy(G0)
    # igraph
    g = ig.Graph(directed=False)
    g.add_vertices(list(G.nodes))
    g.add_edges(list(G.edges))
    # 找到最大团
    gcc0 = len(max(g.connected_components(), key=len))
    gcc = gcc0
    r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
    removed_nodes = []  # 如果需要记录removed_nodes则需要对G同时做瓦解
    qc=0 # qc的值
    while(gcc>p_n):
        # igraph
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
        removed_nodes.append(node_with_CI_max_G)
        qc+=1
        g.delete_vertices(node_with_CI_max_g)
        gcc = len(max(g.connected_components(), key=len))
        r_list.append(gcc/len(G0))
    # 计算 R 值
    Rc_CI = sum(r_list)
    return Rc_CI,r_list,removed_nodes

def HBA_with_seeds(G0,p_n=1):
    G=copy.deepcopy(G0)
    g = ig.Graph(directed=False)
    g.add_vertices(list(G.nodes))
    try:
        g.add_edges(list(G.edges))
    except:
        print(1)
    # 找到最大团
    gcc0 = len(max(g.connected_components(), key=len))
    gcc = gcc0
    r_list = [gcc/len(G0)] # 曲线初始截距 /len(G): 标准化
    removed_nodes = []
    qc=0 # qc的值
    while(gcc>p_n):
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
        removed_nodes.append(node_with_bet_max_G)
        g.delete_vertices(node_with_bet_max_g)
        qc+=1
        gcc = len(max(g.connected_components(), key=len)) # 当前最大连通子团
        r_list.append(gcc/len(G0))
    # 计算 R 值
    Rc_bet = sum(r_list)
    return Rc_bet,r_list,removed_nodes


if __name__=="__main__":
    print('hi')

