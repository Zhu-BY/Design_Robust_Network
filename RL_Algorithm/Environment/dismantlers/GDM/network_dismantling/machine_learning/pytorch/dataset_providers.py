#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

from glob import glob

from pathlib2 import Path

from network_dismantling.machine_learning.pytorch.common import load_graph


# def storage_provider(location, filter="*", callback=None):
#     networks = list()
#     for file in sorted(glob(str(location / (filter + ".graphml")))):
#         filename = Path(file).stem

#         network = load_graph(file)

#         assert not network.is_directed()

#         network.graph_properties["filename"] = network.new_graph_property("string", filename)

#         if callback:
#             callback(filename, network)

#         networks.append((filename, network))

#     return networks
def storage_provider(location, filter="*", callback=None,id=0):
    networks = list()
    # for file in sorted(glob(str(location / (filter + ".graphml")))):
    file = str(location/Path("graph%s.graphml"%id))
    filename = Path(file).stem

    network = load_graph(file)

    assert not network.is_directed()

    network.graph_properties["filename"] = network.new_graph_property("string", filename)

    if callback:
        callback(filename, network)

    networks.append((filename, network))

    return networks

# 修改以适配networkx.graph
from graph_tool.all import *
import networkx as nx
def storage_provider_nx(G_nx, location, filter="*", callback=None):

    networks = list()
    for file in sorted(glob(str(location / (filter + ".graphml")))):
        filename = Path(file).stem

        network = load_graph(file)

        assert not network.is_directed()

        network.graph_properties["filename"] = network.new_graph_property("string", filename)
        calculate_properties(network)
        if callback:
            callback(filename, network)

        networks.append((filename, network))
    

    G_gt = Graph(directed=False)
    # 添加节点
    node_map = {}
    for node in G_nx.nodes():
        v = G_gt.add_vertex()
        node_map[node] = v  # 记录 NetworkX ID 对应的 graph-tool ID
    # 添加边
    for u, v in G_nx.edges():
        G_gt.add_edge(node_map[u], node_map[v])

    network_ = G_gt

    assert not network_.is_directed()

    # network.graph_properties["filename"] = network.new_graph_property("string", filename)

    # # 计算并添加顶点属性
    # chi_degree = network_.new_vertex_property("int")
    # clustering_coefficient = network_.new_vertex_property("float")
    # degree = network_.new_vertex_property("int")
    # kcore = network_.new_vertex_property("int")
    # static_id = network_.new_vertex_property("int")
    # _graphml_edge_id = network_.new_vertex_property("int")
    # # 计算 chi_degree, clustering_coefficient, degree, kcore
    # for v in network_.vertices():
    #     chi_degree[v] = network_.vertex(v).out_degree()  # 示例计算
    #     clustering_coefficient[v] = local_clustering(network_, v)
    #     degree[v] = network_.vertex(v).out_degree()
    #     kcore[v] = kcore_decomposition(network_)[v]
    #     static_id[v] = int(v)
    #     _graphml_edge_id[v] = int(v)
    # # 将属性添加到图中
    # network_.vertex_properties["chi_degree"] = chi_degree
    # network_.vertex_properties["clustering_coefficient"] = clustering_coefficient
    # network_.vertex_properties["degree"] = degree
    # network_.vertex_properties["kcore"] = kcore
    # network_.vertex_properties["static_id"] = static_id
    # network_.vertex_properties["_graphml_edge_id"] = _graphml_edge_id
    return network_

import copy
import numpy as np
def calculate_properties(g0):
    g=copy.deepcopy(g0)
    # remove_edge_list = [e for e in g.edges() if e.source() == e.target()]
    # for e in remove_edge_list:
    #     g.remove_edge(e)
    # 计算度数
    degree_prop = g.degree_property_map("total")
    # 归一化函数（Min-Max 归一化）
    def normalize_property(prop):
        values = [prop[v] for v in g.vertices()]
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return prop  # 如果所有值相同，不做归一化
        norm_prop = g.new_vertex_property("double")
        for v in g.vertices():
            # norm_prop[v] = (prop[v] - min_val) / (max_val - min_val)
            norm_prop[v] = prop[v]/ max_val
        return norm_prop
    # 对 degree 进行归一化
    degree_norm = normalize_property(degree_prop)
    average_degree = np.mean(degree_norm)
    # 计算 chi-degree（基于归一化后的 degree）
    chi_degree = g.new_vertex_property("double")
    for v in g.vertices():
        # neighbor_degrees_norm = [degree_norm[n] for n in v.all_neighbors()] 
        # # neighbor_degrees_norm = [degree_prop[n] for n in v.all_neighbors() if n != v]  # 仅保留非自环邻居
        # # neighbor_degrees_norm = [x/max(neighbor_degrees) for x in neighbor_degrees]
        # if len(neighbor_degrees_norm) > 0:
        #     mean_d_norm = np.mean(neighbor_degrees_norm)  # 计算均值 E[d_norm]
        #     std_d_norm = np.std(neighbor_degrees_norm)    # 计算标准差 σ_d_norm
        #     chi_degree[v] = (mean_d_norm - std_d_norm) ** 2 / mean_d_norm if mean_d_norm > 0 else 0
        # else:
        #     chi_degree[v] = 0  # 没有邻居的情况设为0
        chi_degree[v] = (degree_norm[v] - average_degree) ** 2 / average_degree

    # 计算 clustering coefficient 和 k-core
    clustering_coeff = local_clustering(g)
    kcore_prop = kcore_decomposition(g)

    # 归一化 clustering coefficient 和 k-core
    # g.vertex_properties["clustering_coefficient"] = clustering_coeff
    # g.vertex_properties["kcore"] = kcore_prop
    # g.vertex_properties["degree_norm"] = degree_norm
    g.vertex_properties["degree"] = degree_norm
    g.vertex_properties["clustering_coefficient"] = normalize_property(clustering_coeff)
    g.vertex_properties["kcore"] = normalize_property(kcore_prop)
    g.vertex_properties["chi_degree"] = chi_degree
    # # 对 chi-degree 进行归一化
    # chi_degree_norm = normalize_property(chi_degree)
    # g.vertex_properties["chi_degree_norm"] = chi_degree_norm
    def z_score_normalize_property(prop):
        values = [prop[v] for v in g.vertices()]
        mean_val, std_val = np.mean(values), np.std(values)
        if std_val == 0:
            return prop  # 如果标准差为0，不做归一化
        norm_prop = g.new_vertex_property("double")
        for v in g.vertices():
            norm_prop[v] = (prop[v] - mean_val) / std_val
        return norm_prop
    return g