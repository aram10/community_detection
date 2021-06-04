import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import random
import pickle
import itertools

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.linalg import diag
from func_timeout import func_timeout, FunctionTimedOut
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.generators.community import LFR_benchmark_graph
from itertools import count


def generate_LFR_network(num_vertices, avg_degree):
    return nx.LFR_benchmark_graph(num_vertices, 3, 1.5, 0.1, avg_degree, min_community=50, max_iters=5000)

def create_adjacency_matrix(graph):
    return np.asarray(nx.adjacency_matrix(graph).todense())

def create_pairwise_community_indicator_matrix(community_list):
    num_vertices = len(community_list)
    H = np.zeros(shape=(num_vertices, num_vertices))
    for community in community_list:
        for pair in itertools.combinations_with_replacement(community_list, 2):
            H[pair[0],pair[1]] = 1
    return H

def create_laplacian_matrix(ci_matrix):
    return tf.linalg.diag(tf.reduce_sum(ci_matrix, 1)) - ci_matrix

def combine_data(B, L):
    return np.concatenate((B,L), axis=1)

def get_community_list(graph, name):
    com_list = list({frozenset(graph.nodes[v][name]) for v in graph})
    communities = [list(com_list[x]) for x in range(0, len(com_list))]
    return communities
    
def get_community_labels(graph, name):
    communities = get_community_list(graph, name)
    labels = np.zeros(len(graph), dtype=int)
    x = 0
    for community in communities:
        for vertex in community:
            labels[vertex] = x
        x += 1
    return labels

def get_num_communities(graph, name):
    return len(list({frozenset(graph.nodes[v][name]) for v in graph}))

def reconstruct_communities(labels):
    communities = {}
    for x in range(max(labels)+1):
        communities[x] = []
    i = 0
    for x in labels:
        communities[x].append(i)
        i += 1
    return communities

def plot_communities(graph, name):
    groups = set(nx.get_node_attributes(graph,name).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = graph.nodes()
    colors = [mapping[graph.node[n][name]] for n in nodes]

    pos = nx.spring_layout(graph)
    ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, 
                            with_labels=False, node_size=100, cmap=plt.cm.jet)

def karate_club_communities(graph):
    clublist = [(graph.nodes[v]['club']) for v in graph]
    communities = np.zeros(len(graph), dtype=int)
    x = 0
    for el in clublist:
        if el == 'Officer':
            communities[x] = 1
        x += 1
    return communities