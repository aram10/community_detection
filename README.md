# Community Detection with Autoencoders

This repository contains code for using autoencoders to perform static and dynamic community detection, and Jupyter notebooks with use cases. While the autoencoder was designed with community detection in mind, the framework is still general enough to be used in most contexts requiring an autoencoder. The primary source files of this repository are:

+ `Autoencoder.py`: TensorFlow implementation of an autoencoder. Can be initialized with 1 or more hidden layers. If multiple layers, trained using Hinton's greedy unsupervised layer-wise training method. Can be made sparse with `SparseRegularizer` (KL-divergence) applied to hidden layer(s). Required/optional parameters are detailed in the file comments.
+ `helpers.py`: Helper methods for manipulating graphs/graph representations and community labelings.

Graph objects and supplemental files in `/graphs`. All graphs are stored using `networkx`, and graph objects are serialized with `pickle`. Trained models are stored in `/trained`, with descriptive-enough names that they should be self-explanatory (e.g., cora_L2_sparse_5000iter_1 is the first of a sparse three-model stack, trained with L2 weight regularization for 5000 full-batch epochs).

The rest of this repository is Jupyter notebooks featuring community detection on a variety of graphs, static/dynamic and synthetic/real. 

