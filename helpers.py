import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import random
import pickle
import itertools
import math

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

#see "Incorporating network structure with node contents for community detection on large networks using deep learning"
def adjacency_to_similarity(A):
    S = np.zeros(shape=np.shape(A))
    n = np.shape(A)[0]
    for i in range(n):
        for j in range(n):
            row_i = A[i]
            row_j = A[j]
            S[i,j] = 2 * np.sum(np.bitwise_and(row_i, row_j)) / (np.sum(row_i) + np.sum(row_j))
            
def cora_labels(graph):
    n = len(graph)
    communities_list = list(frozenset((graph.nodes[node]['gt']) for node in graph.nodes))
    communities = {}
    for x in range(len(communities_list)):
        communities[communities_list[x]] = x
    labels = []
    for node in graph.nodes:
        labels.append(communities[graph.nodes[node]['gt']])
    labels = np.asarray(labels, dtype=int)
    return labels

def create_adjacency_matrix(graph):
    return np.asarray(nx.adjacency_matrix(graph).todense())

def create_degree_matrix(X):
    return np.diag(np.sum(X, axis=1))

def create_laplacian(A):
    return create_degree_matrix(A) - A

def create_normalized_laplacian(A):
    func = np.vectorize(lambda i : 1/(math.sqrt(i)))
    D = create_degree_matrix(A)
    D2 = np.diag(func(D.diagonal()))
    return np.matmul(D2, np.matmul(A, D2))

def create_pairwise_community_indicator_matrix(community_list):
    num_vertices = len([v for sublist in community_list for v in sublist])
    H = np.zeros(shape=(num_vertices, num_vertices))
    for community in community_list:
        for pair in itertools.combinations_with_replacement(community, 2):
            H[pair[0],pair[1]] = 1
    return H

def generate_LFR_network(num_vertices, mu, avg_degree, max_deg, min_c, max_c, max_i):
    return nx.LFR_benchmark_graph(num_vertices, 2, 1.2, mu, avg_degree, max_degree=max_deg, min_community=min_c, max_community=max_c, max_iters=max_i)

def get_community_labels(graph, name):
    communities = get_community_list(graph, name)
    labels = np.zeros(len(graph), dtype=int)
    x = 0
    for community in communities:
        for vertex in community:
            labels[vertex] = x
        x += 1
    return labels

def get_community_list(graph, name):
    com_list = list({frozenset(graph.nodes[v][name]) for v in graph})
    communities = [list(com_list[x]) for x in range(0, len(com_list))]
    return communities

def get_num_communities(graph, name):
    return len(list({frozenset(graph.nodes[v][name]) for v in graph}))

def karate_club_labels(graph):
    clublist = [(graph.nodes[v]['club']) for v in graph]
    communities = np.zeros(len(graph), dtype=int)
    x = 0
    for el in clublist:
        if el == 'Officer':
            communities[x] = 1
        x += 1
    return communities

def karate_club_communities(graph):
    c1 = []
    c2 = []
    for x in range(len(graph)):
        if graph.nodes[x]['club'] == 'Officer':
            c1.append(x)
        else:
            c2.append(x)
    return [c1, c2]

def plot_communities(graph, name):
    groups = set(nx.get_node_attributes(graph,name).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = graph.nodes()
    colors = [mapping[graph.node[n][name]] for n in nodes]

    pos = nx.spring_layout(graph)
    ec = nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colors, 
                            with_labels=False, node_size=100, cmap=plt.cm.jet)
    
def polbooks_labels(graph):
    n = len(graph)
    communities = {'l': 0, 'n': 1, 'c': 2}
    labels = []
    for node in graph.nodes:
        labels.append(communities[graph.nodes[node]['value']])
    labels = np.asarray(labels, dtype=int)
    return labels

def reconstruct_communities(labels):
    communities = {}
    for x in range(max(labels)+1):
        communities[x] = []
    i = 0
    for x in labels:
        communities[x].append(i)
        i += 1
    for x in communities:
        communities[x].sort()
    return communities
