import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import networkx as nx
import random
import pickle
import itertools
import math
import urllib.request as urllib
import io
import zipfile

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

from helpers import *
from Autoencoder import *

cora = nx.read_gml("cora.gml")

ae_cora_1 = Autoencoder(2708, 512, k_reg=tf.keras.regularizers.L2(), act_reg=SparseRegularizer())
history_cora_1 = train(ae_cora_1, 50000, 2708, X_cora_1)

X_cora_2 = ae_cora_1.encoder(X_cora_1)

ae_cora_2 = Autoencoder(512, 256, k_reg=tf.keras.regularizers.L2(), act_reg=SparseRegularizer())
history_cora_2 = train(ae_cora_2, 50000, 512, X_cora_2)

X_cora_3 = ae_cora_2.encoder(X_cora_2)

ae_cora_3 = Autoencoder(256, 128, k_reg=tf.keras.regularizers.L2(), act_reg=SparseRegularizer())
history_cora_3 = train(ae_cora_3, 50000, 256, X_cora_3)

ae_cora_1.save('ae_cora_1')
ae_cora_2.save('ae_cora_2')
ae_cora_3.save('ae_cora_3')