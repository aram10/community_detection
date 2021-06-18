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
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.generators.community import LFR_benchmark_graph
from itertools import count
from scipy.spatial import distance_matrix

#A = [a]_ij, a_ij = 0 ==>  b_ij = 1, a_ij = 1 ==> b_ij = beta > 1
#Used for weighting reconstruction loss, see "Structural Deep Network Embedding" (Wang et. al, 2016)
def adjacency_to_loss(A, beta):
    B = np.empty(shape=np.shape(A))
    B.fill(1)
    for r in range(A.shape[0]):
        for c in range(A.shape[1]):
            if A[r,c] == 1:
                B[r,c] = beta
    return B

#Sørensen–Dice similarity matrix
def adjacency_to_similarity(A):
    S = np.zeros(shape=np.shape(A))
    n = np.shape(A)[0]
    for i in range(n):
        for j in range(n):
            row_i = A[i]
            row_j = A[j]
            S[i,j] = 2 * np.sum(np.bitwise_and(row_i, row_j)) / (np.sum(row_i) + np.sum(row_j))
    return S
            
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

#A~ = A + I
def create_adjacency_matrix_with_self_connections(graph):
    A = create_adjacency_matrix(graph)
    return np.add(A, np.identity(A.shape[0]))

#parameters: numpy array 'labels,' where labels[i] is the community membership of the ith node
#returns: 2D numpy array C where C[i,j] is 1 if nodes i and j belong to the same community and 0 otherwise
def create_pairwise_community_membership_matrix(labels):
    n = labels.shape[0]
    C = np.identity(n)
    for i in range(1, n):
        for j in range(i):
            if labels[i] == labels[j]:
                C[i,j] = 1
    return symmetrize(C)

def create_degree_matrix(X):
    return np.diag(np.sum(X, axis=1))

def create_laplacian(A):
    return create_degree_matrix(A) - A

def create_normalized_laplacian(A):
    func = np.vectorize(lambda i : 1/(math.sqrt(i)))
    D = create_degree_matrix(A)
    D2 = np.diag(func(D.diagonal()))
    return np.matmul(D2, np.matmul(create_laplacian(A), D2)).astype(np.float32)

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

#input: kxn matrix H
#returns: 1xn matrix 'labels,' where labels[i] = argmax_{k} H[k,i]
def nmf_cluster_membership(H):
    n = H.shape[1]
    labels = np.ndarray(n)
    for i in range(n):
        labels[i] = np.argmax(H[:,i])
    return labels

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

def probability_transition_matrix(A, k):
    D = create_degree_matrix(A)
    P = np.matmul(np.linalg.inv(D), A)
    for x in range(k-1):
        P = np.matmul(P, P)
    return P.astype('float32')

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

#See Equation (2) in "Stacked autoencoder-based community detection method via an ensemble clustering framework" (Xu et al., 2020)
def similarity_matrix_2(graph, kp):
    B = nx.modularity_matrix(graph).astype('float32')
    E = distance_matrix(B, B)
    m = np.mean(np.mean(B, axis=0))
    return np.exp(np.negative(np.divide(E, kp*m)))
    
def symmetrize(a):
    return a + a.T - np.diag(a.diagonal())
