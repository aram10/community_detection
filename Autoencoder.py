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

class Autoencoder(Model):
    
    def __init__(self, n, latentdim):
        super(Autoencoder, self).__init__()
        self.n = n
        self.encoder = tf.keras.Sequential([
            layers.Dense(latentdim, activation='relu', activity_regularizer=tf.keras.regularizers.L1(10e-5))
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(n, activation='sigmoid')
        ])
        
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded