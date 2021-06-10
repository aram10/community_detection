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
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.linalg import diag
from func_timeout import func_timeout, FunctionTimedOut
from tensorflow.keras import callbacks
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from networkx.generators.community import LFR_benchmark_graph
from itertools import count

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=500,
    decay_rate=0.96,
    staircase=True)

class SparseRegularizer(keras.regularizers.Regularizer):
    
    def __init__(self, rho = 0.01,beta = 1):
        """
        rho  : Desired average activation of the hidden units
        beta : Weight of sparsity penalty term
        """
        self.rho = rho
        self.beta = beta
        
    def __call__(self, activation):
        rho = self.rho
        beta = self.beta
        activation = tf.nn.sigmoid(activation)
        rho_bar = K.mean(activation, axis=0)
        rho_bar = K.maximum(rho_bar,1e-10) 
        KLs = rho*K.log(rho/rho_bar) + (1-rho)*K.log((1-rho)/(1-rho_bar))
        return beta * K.sum(KLs)
    
    def get_config(self):
        return {
            'rho': self.rho,
            'beta': self.beta
        }

class Autoencoder(Model):
    
    def __init__(self, n, latentdim, lam=0.5, activ='sigmoid', loss=tf.keras.losses.MeanSquaredError(), k_reg=None, sparse=False):
        super(Autoencoder, self).__init__()
        self.n = n
        self.latentdim = latentdim
        self.lam = lam
        self.loss = loss
        self.k_reg = k_reg
        self.sparse = sparse
        self.encoder = tf.keras.Sequential()
        if sparse:
            self.encoder.add(layers.Dense(latentdim, activation=activ, kernel_regularizer=k_reg, activity_regularizer=SparseRegularizer()))
        else:
            self.encoder.add(layers.Dense(latentdim, activation=activ, kernel_regularizer=k_reg))
        self.decoder = tf.keras.Sequential([
            layers.Dense(n, activation='sigmoid')
        ])
        
    def call(self, X):
        encoded = self.encoder(X)
        decoded = self.decoder(encoded)
        return decoded

def loss(model, X):
    H = model.encoder(X)
    X_=model.decoder(H)
    return model.loss(X, X_)
    
def train_step(loss, model, original, opt=tf.keras.optimizers.Adam(learning_rate=lr_schedule)):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)
        
def train(model, epochs, batch_size, data):
    training_dataset = tf.data.Dataset.from_tensor_slices(data)
    training_dataset = training_dataset.batch(batch_size)
    training_dataset = training_dataset.shuffle(data.shape[0])
    training_dataset = training_dataset.prefetch(batch_size * 4)
    for epoch in range(epochs):
        for step, batch_features in enumerate(training_dataset):
            train_step(loss, model, batch_features)