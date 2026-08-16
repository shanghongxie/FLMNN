#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 13:12:46 2026

@author: shanghongxie
"""

# Import the simulation-data module and its main generator function.

from genSimData_flmnn import generate_dense_data


# Import the FLMNN module and the user-facing FLMNN estimator class.
from flmnn import FLMNN

# Generate dense and regular multidimensional functional data
data = generate_dense_data(
    Ns=(4, 4, 16),   # Number of observation grid points in each
                        # functional dimension.
                        # Each functional observation is evaluated on
                        # a 10 x 10 x 16 grid.

    Ms=(4, 4, 4),      # Number of marginal B-spline basis functions
                        # used in each of the three dimensions.

    npc=10,            # Number of true functional principal components
                        # used to generate the simulated functional data.

    Ntrain=100,        # Number of simulated training subjects.

    Ntest=1000,         # Number of simulated test subjects.

    sigma2x=1,         # Variance of measurement noise added to the
                        # simulated functional observations.

    sigma2y=1,         # Variance of noise added in the mechanism used
                        # to generate the classification labels.

    random_state=1             # Random seed for reproducibility.
)

# Observed functional data with noise in training data
train_Y = data["Y_noisy"]
# Class labels for the training sample
trainlabel = data["trainlabel"]

# Observed functional data with noise in test data
test_Y = data["test_Y_noisy"]
testlabel = data["testlabel"]

# True eigenfunctions
true_eigenfunctions = data["true_eigenfunctions"]
eigenvalues = data["eigenvalues"]

# True FPC scores
train_true_scores = data["train_true_scores"]
test_true_scores = data["test_true_scores"]



# Evaluation
model = FLMNN(
    Ms=(4, 4, 4), # number of marginal basis for each dimension
    K=30, # Rank of the marginal product basis
    lambda_d1_grid=(1e-11, 1e-10), #Candidate marginal roughness penalties applied to the first D-1 functional dimensions.
    lambda_d2_grid=(1e-11, 1e-10), # Candidate marginal roughness penalties applied to the final
   # functional dimension.
    lambda_b_grid=(1e-8,), # Candidate regularization parameters for the subject coefficient
   # matrix in MARGARITA.
    neighbors=(5,), # Candidate numbers of neighbors 
    mus=(0.5,), # Candidate LMNN regularization parameters.
    Ks=(10,), # Candidate numbers of FPC scores 
)


model.fit(train_Y, trainlabel)

testpred = model.predict(test_Y)

import numpy as np
testacc = np.mean(testpred == testlabel)

print("Test accuracy:", testacc)


mahalanobis_matrix = model.mahalanobis_matrix_

opt_lambda_d1 = model.lambda_d1_
opt_lambda_d2 = model.lambda_d2_
