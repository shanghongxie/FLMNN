#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:47:34 2024

@author: shanghongxie
"""

import numpy as np 
import tensorly as tl
import sys
from tensor_decomposition import MARGARITA
from marginal_product_basis import MPB
from utility import FPCA 
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSpline, Tensor
from skfda.misc.operators import LinearDifferentialOperator, gram_matrix
from skfda.preprocessing.smoothing import BasisSmoother
from skfda import FDataGrid

from sklearn.metrics import r2_score, mean_squared_error
from collections import namedtuple 
import time 
import os 
import itertools 
import pickle 
import scipy 
from scipy.stats import ortho_group, sem
import pandas as pd 
import functools 
import operator 

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns


svdtuple = namedtuple("SVD", ["U", "s", "Vt"])


## Wrapper function for K-oMPB estimation using MARGARITA [1]
def kompb(Y, K, Phis, nmode):
    ## Compute basis evaluation matrices and SVDs
    Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
    ## Specify differential operator for penalization 
    D2 = LinearDifferentialOperator(2)
    Rlst = [gram_matrix(D2, bspline_basis[d]) for d in range(nmode)] 
    ## Perform the n-mode coordinate transformations into the spline coefficient space 
    ## Y_Bar should be close to zero so no need to center 
    G = tl.tenalg.multi_mode_dot(Y, [svdt.U.T for svdt in Svds], list(range(nmode)))
    ## Estimation MPB
    maxiter = (200, 100)
    tol_inner = (1e-3, 1e-3)
    tol_outer = 1e-3
    initialize = "random"
    Vs = [Svds[d].Vt.T for d in range(nmode)]
    Dinvs = [np.diag(1./Svds[d].s) for d in range(nmode)]
    Tlst_bcd = [Dinvs[d]@Vs[d].T@Rlst[d]@Vs[d]@Dinvs[d] for d in range(nmode)]
    pen_params = (1e-10, 1e-10, 1e-10) ## no noise in simulation so regularization can be mild 
    start = time.time()
    Ctilde, Smat, scalars, FLAG_C, FLAG_N = MARGARITA(G, Tlst_bcd, pen_params, K, 
                                     max_iter=maxiter, tol_inner=tol_inner, 
                                     tol_outer=tol_outer,  regularization="l2", init=initialize, 
                                    verbose=False)
    elapsed = time.time() - start
    Clst = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Ctilde[d] for d in range(nmode)] 
    Smat_scaled = np.multiply(Smat, scalars)
    Zeta_tensor = np.zeros((K,n1,n2))
    for k in range(K):
        Zeta_tensor[k,:,:] = (Phis[0] @ Clst[0][:,k]).reshape(-1,1) @ (Phis[1] @ Clst[1][:,k]).reshape(-1,1).T
    return Zeta_tensor, Clst, Smat_scaled 

## Wrapper function for two stage FPCA from [1]
def two_stage_fpca(Zeta_tensor, Clst, Smat_scaled, K):
    mpb = MPB(bspline_basis, Clst)
    ## Perform FPCA 
    J = mpb.gram_matrix()
    R = mpb.roughness_matrix()
    B, gamma = FPCA(Smat_scaled, J, R, lam=1e-10)
    Eta_tensor = tl.tenalg.mode_dot(Zeta_tensor, B.T, 0)
    return Eta_tensor
    
## Implementation of marginal product FPCA from [2]
def marginal_product_fpca(Y, K):
    ## Step 1: Obtain pre-smoothed estimates 
    N = Y.shape[-1]
    tpb_smoother = BasisSmoother(tp_basis, return_basis=True)
    fd = FDataGrid(np.moveaxis(Y, [0, 1, 2], [1, 2, 0]), xgrids)
    fd_smooth = tpb_smoother.fit_transform(fd)
    Coefs_hat = fd_smooth.coefficients.reshape(N, m1, m2) ## hopefully this is the correct reshaping 

    fd_mu = FDataGrid(np.mean(Y, axis=2).reshape(1, n1,n2), xgrids)
    fd_smooth_mu = tpb_smoother.fit_transform(fd_mu)
    coefs_mu = fd_smooth_mu.coefficients.ravel().reshape(m1, m2)

    ## Step 2: Compute both marginal covariances 
    Jlst = [bsp1.gram_matrix(), bsp2.gram_matrix()]
    G_x1 = np.zeros((m1,m1))
    G_x2 = np.zeros((m2, m2))

    for n in range(N):
        G_x1 = G_x1 + Coefs_hat[n,:,:] @ Jlst[1] @ Coefs_hat[n,:,:].T
        G_x2 = G_x2 + Coefs_hat[n,:,:].T @ Jlst[0] @  Coefs_hat[n,:,:]

    G_x1 = G_x1/N
    G_x2 = G_x2/N

    ## Step 3: Perform FPCA using standard methods 
    Eigenlst = [np.linalg.eig(Jlst[0]), np.linalg.eig(Jlst[1])]
    Jsqrt_lst = [Eigenlst[0][1] @ np.diag(np.sqrt(Eigenlst[0][0])) @ Eigenlst[0][1].T, 
                 Eigenlst[1][1] @ np.diag(np.sqrt(Eigenlst[1][0])) @ Eigenlst[1][1].T]
    Jinvsqrt_lst = [Eigenlst[0][1] @ np.diag(1/np.sqrt(Eigenlst[0][0])) @ Eigenlst[0][1].T, 
                 Eigenlst[1][1] @ np.diag(1/np.sqrt(Eigenlst[1][0])) @ Eigenlst[1][1].T]

    fpca_x1 = np.linalg.eig(Jsqrt_lst[0] @ G_x1 @ Jsqrt_lst[0])
    fpca_x2 =  np.linalg.eig(Jsqrt_lst[1] @ G_x2 @ Jsqrt_lst[1])

    eige_coefs_x1 = Jinvsqrt_lst[0] @ fpca_x1[1]
    eige_coefs_x2 = Jinvsqrt_lst[1] @ fpca_x2[1]

    eigenvalues = fpca_x1[0].reshape(-1,1) @ fpca_x2[0].reshape(1,-1)
    eigenvalues_orderd = -np.sort(-eigenvalues.ravel())
    ordering = []
    for ev in eigenvalues_orderd:
        xo, yo = np.where(eigenvalues == ev)
        ordering.append((xo[0],yo[0]))

    ## Step 4: Create regression matrix 
    MFPCA_tensor = np.zeros((K,n1,n2))
    for k in range(K):
        k1, k2 = ordering[k]
        MFPCA_tensor[k, :, :] = (Phis[0] @ eige_coefs_x1[:,k1].reshape(-1,1)) @ (Phis[1] @ eige_coefs_x2[:,k2].reshape(-1,1)).T
        
    return MFPCA_tensor

def non_separable_bias(Coefs, lambdas, Clst):
    K = Clst[0].shape[1]
    Jlst = [bsp1.gram_matrix(), bsp2.gram_matrix()]
    JC0 = Jlst[0] @ Clst[0]
    JC1 = Clst[1].T @ Jlst[1]
    V = np.block([np.kron(JC1[kk,:].reshape(1,-1).T, JC0[:,kk].reshape(-1,1)) for kk in range(K)])
    NSB = 0 
    for k in range(K):
        rho_k = lambdas[k]
        A_k = Coefs[k,].reshape(m1, m2)
        JAJ_k = Jlst[0] @ A_k @ Jlst[1]
        a_k = JAJ_k.reshape((-1, 1), order="F") ## stacking columns 
        b_k, res, rank, s = np.linalg.lstsq(V, a_k, rcond=None)
        Diff = A_k - Clst[0] @ np.diag(b_k.ravel()) @ Clst[1].T
        NSB += rho_k*(np.linalg.norm(Jlst[0] @ Diff @ Jlst[1], ord="fro")**2)
    return NSB