#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 08:58:38 2026

@author: shanghongxie
"""


# Import the simulation-data module and its main generator function.

from genData import generate_dense_data


# Import the FLMNN module and the user-facing FLMNN estimator class.
from flmnn import FLMNN



# ----------------------------------------------------------
# Scenario 1: Dense and regular functional predictors
# ----------------------------------------------------------

# Generate simulated multidimensional functional data
data = generate_dense_data(
    Ns=(10, 10, 64),   # Number of observation grid points in each
                        # functional dimension.
                        # Each functional observation is evaluated on
                        # a 10 x 10 x 64 grid.

    Ms=(5, 5, 8),      # Number of marginal B-spline basis functions
                        # used in each of the three dimensions.

    npc=10,            # Number of true functional principal components
                        # used to generate the simulated functional data.

    Ntrain=100,        # Number of simulated training subjects.

    Ntest=1000,         # Number of simulated test subjects.

    sigma2x=1,         # Variance of measurement noise added to the
                        # simulated functional observations.

    sigma2y=5,         # Variance of noise added in the mechanism used
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
    Ms=(5, 5, 8), # number of marginal basis for each dimension
    K=30, # Rank of the marginal product basis
    lambda_d1_grid=(1e-11, 1e-10, 1e-8), #Candidate marginal roughness penalties applied to the first D-1 functional dimensions.
    lambda_d2_grid=(1e-11, 1e-10, 1e-8), # Candidate marginal roughness penalties applied to the final
   # functional dimension.
    lambda_b_grid=(1e-10, 1e-8, 1e-6), # Candidate regularization parameters for the subject coefficient
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

estimated_eigenfunctions = model.fpca_eigenfunctions_

estimated_eigenvalues = model.fpca_eigenvalues_


## plot eigenfunctions
from plotting import plot_eigenfunction_slice

plot_eigenfunction_slice(
    estimated_eigenfunctions,
    k=0,
    third_idx=5,
    true_eigenfunctions=true_eigenfunctions # optional
)




# ----------------------------------------------------------
# Scenario 2: Sparse and irregular functional predictors
# ----------------------------------------------------------
    

from genData import generate_sparse_data


data = generate_sparse_data(
    Ns=(10, 10, 64),
    Ms=(5, 5, 8),  
    npc=10, 
    Ntrain=50,
    Ntest=1000,
    sigma2x=5,
    sigma2y=0.5,
    n_locations=20,
    random_state=1,
)



trainlabel = data["trainlabel"]
testlabel = data["testlabel"]

train_sparse_Y = data["train_sparse_Y"]
test_sparse_Y = data["test_sparse_Y"]


from genData import interpolate_sparse_data

train_Y_interp = interpolate_sparse_data(
    train_sparse_Y,
    data["train_mask"],
)

test_Y_interp = interpolate_sparse_data(
    test_sparse_Y,
    data["test_mask"],
)

# Evaluation
model = FLMNN(
    Ms=(5, 5, 8), # number of marginal basis for each dimension
    K=30, # Rank of the marginal product basis
    lambda_d1_grid=(1e-11, 1e-10, 1e-8), #Candidate marginal roughness penalties applied to the first D-1 functional dimensions.
    lambda_d2_grid=(1e-11, 1e-10, 1e-8), # Candidate marginal roughness penalties applied to the final
   # functional dimension.
    lambda_b_grid=(1e-10, 1e-8, 1e-6), # Candidate regularization parameters for the subject coefficient
   # matrix in MARGARITA.
    neighbors=(5,), # Candidate numbers of neighbors 
    mus=(0.5,), # Candidate LMNN regularization parameters.
    Ks=(10,), # Candidate numbers of FPC scores 
)


model.fit(train_Y_interp, trainlabel)

testpred = model.predict(test_Y_interp)

import numpy as np
testacc = np.mean(testpred == testlabel)

print("Test accuracy:", testacc)
