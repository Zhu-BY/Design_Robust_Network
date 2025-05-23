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

from collections import defaultdict
from itertools import combinations, chain

import numpy as np
from graph_tool.all import *

from network_dismantling.machine_learning.pytorch.common import chi
from network_dismantling.machine_learning.pytorch.config import all_features as features_list


def training_data_extractor(g, threshold=None, output_file=None, target_property_name="target", vertices=None,
                            compute_targets=True, features=None, logger=None, k_range=None):
    if logger is None:
        def logger(record):
            pass
    # def logger(record):
    #     log_queue.put(record)
    #     pass

    if features is None:
        features_dict = defaultdict(lambda: True)
    else:
        features_dict = defaultdict(lambda: False, ((f, True) for f in features))

    if vertices is None:
        vertices = g.get_vertices()

    out_features = dict()

    if features_dict["degree"] or features_dict["chi_degree"]:
        # logger("Computing Degree")
        degree = g.get_out_degrees(vertices)

        # logger("Computing Max degree")
        max_degree = np.max(degree)

        # logger("Normalizing the degree")
        degree = np.divide(degree, max_degree)

        if features_dict["degree"]:
            out_features["degree"] = degree

        if features_dict["chi_degree"]:
            # logger("Computing Mean Normalized Degree")
            average_degree = np.mean(degree)

            # logger("Computing Chi Degree")
            chi_degree = [chi(degree[v], average_degree) for v in vertices]

            out_features["chi_degree"] = chi_degree

    if features_dict["eigenvectors"]:
        # logger("Computing Eigenvectors")
        _, eigenvectors = eigenvector(g)

        out_features["eigenvectors"] = eigenvectors

    if features_dict["clustering_coefficient"] or features_dict["chi_lcc"]:

        # logger("Computing LCC")
        clustering_coefficient = local_clustering(g)

        # logger("Computing Mean LCC")
        avg_lcc = np.mean(clustering_coefficient.get_array())

        # logger("Computing Chi LCC")
        chi_lcc = [chi(clustering_coefficient[v], avg_lcc) for v in vertices]

        if features_dict["clustering_coefficient"]:
            out_features["clustering_coefficient"] = clustering_coefficient
        if features_dict["chi_lcc"]:
            out_features["chi_lcc"] = chi_lcc

    # logger("Computing Neighbors")
    # neighbours = [g.get_out_neighbors(v) for v in vertices]
    #
    # logger("Computing Neighbors Mean Chi Degree")
    # mean_chi_degree = [np.mean([chi_degree[i] for i in x]) if len(x) > 0 else 0 for x in neighbours]
    #
    # logger("Computing Neighbors Mean Chi LCC")
    # mean_chi_lcc = [np.mean([chi_lcc[i] for i in x]) if len(x) > 0 else 0 for x in neighbours]

    if features_dict["pagerank_out"]:
        # logger("Computing Pagerank")
        pagerank_out = pagerank(g)

        out_features["pagerank_out"] = pagerank_out

    if features_dict["betweenness_centrality"]:
        # logger("Computing Betweenness")
        betweenness_centrality, _ = betweenness(g)

        out_features["betweenness_centrality"] = betweenness_centrality

    if features_dict["kcore"]:
        # K-core
        # logger("Computing the K-core and normalising it")
        kcore = kcore_decomposition(g)

        kcore_array = kcore.get_array()
        kcore_array = np.divide(kcore_array, np.max(kcore_array))

        out_features["kcore"] = kcore_array

    # logger("Adding Properties")

    if compute_targets is True:
        targets = g.new_vertex_property("float")
        g.vertex_properties[target_property_name] = targets

        # logger("Computing LCC size deltas")

        best_combinations = set()
        num_vertices = g.num_vertices()

        if threshold is None:
            exit("ERROR: No threshold provided!")

        stop_condition = np.ceil(float(num_vertices * threshold))

        if k_range is None:
            k_range = range(0, num_vertices)
        for k in k_range:
            # logger("Trying with {} / {} vertices long combinations".format(k, num_vertices))
            best_score = num_vertices

            # Generate all the possible combinations of length k
            for combination in combinations(g.get_vertices(), k):
                local_network = g.copy()

                local_network.remove_vertex(combination)

                local_network_lcc_size = (np.count_nonzero(label_largest_component(local_network).get_array()))

                if local_network_lcc_size < best_score:
                    best_score = local_network_lcc_size
                    best_combinations = set()
                    best_combinations.add(combination)
                elif local_network_lcc_size == best_score:
                    best_combinations.add(combination)

            if best_score <= stop_condition:
                # logger("Found that {} vertices break the network apart".format(k))
                break

        num_combinations = len(best_combinations)

        # Count number of occurrences:
        all_occurrences = list(chain.from_iterable(best_combinations))
        unique, counts = np.unique(all_occurrences, return_counts=True)

        if len(unique) == 0:
            # Handle networks that already satisfy the requirement
            for v in vertices:
                targets[v] = 0
        else:

            for key, value in zip(list(unique), list(counts)):
                # occurrences_count
                # TODO CHECK ME!
                targets[key] = (value / num_combinations)

    # Store the features as vertex properties
    # TODO IMPROVE ME
    for f in features_list:
        if features_dict[f] is not True:
            continue

        if f in ["num_vertices", "num_edges"]:
            continue

        property = g.new_vertex_property("float", vals=out_features[f])

        # Add vertex properties to the internal representation of the graph
        g.vertex_properties[f] = property

    # TODO Move me
    if output_file is not None:
        # # Create feature list graph property
        # g.graph_properties["features"] = g.new_graph_property("string", ','.join(
        #     f for f in features_list if features_dict[f] is True))

        # # Create feature vertex property
        # features_property = g.new_vertex_property("string")
        # for v in vertices:
        #     features_property[v] = ','.join(str(out_features[f][v]) for f in features_list if features_dict[f] is True)

        # logger("Storing the graph")
        g.save(output_file, fmt='graphml')