# Spectral-Sparsification-Preserves-Representation-Geometry
Code for Representation Geometry Preservation with Spectral Sparsification in GNNs

This repository contains code for reproducing the experiments in:

**Spectral Graph Sparsification Preserves Representation Geometry in Graph Neural Networks**

The paper studies whether spectral graph sparsification preserves not only graph Laplacian quadratic forms, but also the hidden representation geometry learned by polynomial-filter graph neural networks. The experiments evaluate effective-resistance sparsification on synthetic and real graph datasets, measuring polynomial-filter distortion, hidden representation error, hidden Gram distortion, nearest-neighbor overlap, class-centroid stability, training-trajectory perturbation, and knowledge-distillation behavior.

real_final.py contains experiments on Fashion  MNIST, Cora, and Paul 15 datasets
real_synthetic.py contains experiments on synthetic SBM and geometric graphs.

Copyright Sanjukta Krishnagopal 2026. Please cite appropriately if you use this code.
