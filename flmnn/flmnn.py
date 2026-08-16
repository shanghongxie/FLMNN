#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 09:51:12 2026


@author: shanghongxie
"""



import sys

import numpy as np
import tensorly as tl

from pathlib import Path
import importlib

CODE_DIR = str(Path(__file__).resolve().parent)

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

importlib.invalidate_caches()

import tensor_decomposition
from tensor_decomposition import MARGARITA, fCP_TPA

from marginal_product_basis import MPB
from utility import discrete_laplacian_1D
from utility import FPCA
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.representation.basis import BSpline, Tensor

from skfda.misc.operators import LinearDifferentialOperator, gram_matrix
from skfda import FDataGrid


from collections import namedtuple 
import os 
import itertools 
import scipy 
from scipy.stats import ortho_group, sem
import functools 
import operator 

import argparse

import metric_learn as ml

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning,
)




def sigmoid(x):
    return 1 / (1 + np.exp(-x))


import random


class FLMNN:
    """Functional Large Margin Nearest Neighbor classifier.

    Parameters
    ----------
    Ms : tuple or list of int
        Number of marginal B-spline basis functions for each functional
        dimension.
    K : int
        Rank of the marginal product basis used by MARGARITA.
    lambda_d1_grid : array-like
        Candidate marginal roughness penalties applied to the first D-1
        functional dimensions.
    lambda_d2_grid : array-like
        Candidate marginal roughness penalties applied to the final
        functional dimension.
    lambda_b_grid : array-like
        Candidate regularization parameters for the subject coefficient
        matrix in MARGARITA.
    neighbors : array-like of int
        Candidate numbers of neighbors for LMNN/KNN.
    mus : array-like of float
        Candidate LMNN regularization parameters.
    Ks : array-like of int
        Candidate numbers of FPC scores used for classification.
    nfold : int, default=5
        Number of folds used for both K-oMPB and LMNN cross-validation.
    basis_order : int, default=4
        Order of each marginal B-spline basis.
        
        
    Attributes
    ----------
    mean_function_ : ndarray
        Mean function estimated from the training observations during
        ``fit``. This same mean function is used to center new observations
        during ``transform`` and ``predict``.

    fpca_eigenfunctions_ : ndarray
        Estimated FPCA eigenfunctions obtained from the two-stage FPCA
        procedure. The first axis indexes eigenfunctions and the remaining
        axes correspond to the functional domain.

    mpb_basis_ : ndarray
        Estimated rank-K marginal product basis functions obtained from
        MARGARITA and evaluated on the observation grid.

    fpca_loadings_ : ndarray
        Coefficient matrix defining the estimated FPCA eigenfunctions as
        linear combinations of the marginal product basis functions.

    fpca_eigenvalues_ : ndarray
        Estimated eigenvalues associated with the FPCA eigenfunctions.

    pve_ : ndarray
        Cumulative proportion of variance explained by the estimated
        eigenfunctions.
        
    lambda_d1_ : float
        Value of ``lambda_d1`` selected by cross-validation.

    lambda_d2_ : float
        Value of ``lambda_d2`` selected by cross-validation.

    lambda_b_ : float
        Value of ``lambda_b`` selected by cross-validation.

    train_scores_ : ndarray
        Estimated FPC scores for the training observations, with shape
        ``(n_samples, K)``.

    lmnn_ : metric_learn.LMNN
        Fitted LMNN metric-learning estimator obtained using the selected
        tuning parameters.
    
    knn_ : sklearn.neighbors.KNeighborsClassifier
        Fitted K-nearest-neighbor classifier using the distance metric
        learned by LMNN.

    mahalanobis_matrix_ : ndarray
        Mahalanobis metric matrix learned by LMNN. Its shape is
        ``(opt_K_, opt_K_)``.

    opt_neighbor_ : int
        Number of nearest neighbors selected by cross-validation.

    opt_mu_ : float
        LMNN regularization parameter selected by cross-validation.

    opt_K_ : int
        Number of FPC scores selected for LMNN/KNN classification.

    train_accuracy_ : float
        Classification accuracy of the fitted classifier on the training
        observations.

    train_predictions_ : ndarray
        Predicted class labels for the training observations.

    cv_results_ : dict
        Mean cross-validation accuracy for each candidate combination
        ``(neighbor, mu, K)`` considered during LMNN/KNN tuning.    
    
    classes_ : ndarray
        Unique class labels observed in the training data.    
        
    """

    def __init__(
        self,
        Ms,
        K,
        lambda_d1_grid,
        lambda_d2_grid,
        lambda_b_grid,
        neighbors,
        mus,
        Ks,
        nfold=5,
        basis_order=4,
    ):
        self.Ms = tuple(Ms)
        self.K = int(K)
        self.lambda_d1_grid = tuple(lambda_d1_grid)
        self.lambda_d2_grid = tuple(lambda_d2_grid)
        self.lambda_b_grid = tuple(lambda_b_grid)
        self.neighbors = tuple(neighbors)
        self.mus = tuple(mus)
        self.Ks = tuple(Ks)
        self.nfold = int(nfold)
        self.basis_order = int(basis_order)

    def _check_is_fitted(self):
        if not hasattr(self, "knn_"):
            raise RuntimeError("FLMNN is not fitted. Call fit(train_Y, trainlabel) first.")

    def _validate_grid(self, Y):
        if tuple(Y.shape[:-1]) != self.Ns_:
            raise ValueError(
                "Y must have the same functional grid dimensions as the "
                f"training data. Expected {self.Ns_}, got {tuple(Y.shape[:-1])}."
            )

    def fit(self, train_Y, trainlabel):
        """Fit FLMNN using labeled training functional data.

        Parameters
        ----------
        train_Y : ndarray
            Training tensor with shape ``(n_1, ..., n_D, N_train)``.
        trainlabel : array-like
            Training class labels with shape ``(N_train,)``.

        Returns
        -------
        self : FLMNN
            Fitted estimator.
        """
        train_Y = np.asarray(train_Y)
        trainlabel = np.asarray(trainlabel)

        if train_Y.ndim < 2:
            raise ValueError("train_Y must contain at least one functional dimension and a sample axis.")
        if train_Y.shape[-1] != trainlabel.shape[0]:
            raise ValueError(
                "The number of training samples in train_Y must match the "
                "number of labels in trainlabel."
            )

        self.Ns_ = tuple(train_Y.shape[:-1])
        self.nmode_ = len(self.Ns_)
        self.n_train_ = train_Y.shape[-1]

        if len(self.Ms) != self.nmode_:
            raise ValueError(
                f"Ms must contain one basis dimension per functional mode. "
                f"Expected {self.nmode_}, got {len(self.Ms)}."
            )
        if any(k <= 0 or k > self.K for k in self.Ks):
            raise ValueError("Every candidate in Ks must satisfy 1 <= K_candidate <= K.")

        # Store the training mean and use it for all future score projections.
        self.mean_function_ = train_Y.mean(axis=-1, keepdims=True)

        self.bspline_basis_ = [
            BSpline(n_basis=m, order=self.basis_order)
            for m in self.Ms
        ]
        self.xgrids_ = [
            np.linspace(0, 1, n)
            for n in self.Ns_
        ]

        (
            self.fpca_eigenfunctions_,
            self.mpb_basis_,
            self.fpca_loadings_,
            self.fpca_eigenvalues_,
            self.pve_,
            self.lambda_d1_,
            self.lambda_d2_,
            self.lambda_b_,
            self.converged_,
            self.numerical_warning_,
        ) = fpca_cv(
            train_Y,
            self.K,
            self.bspline_basis_,
            self.xgrids_,
            self.nmode_,
            self.lambda_d1_grid,
            self.lambda_d2_grid,
            self.lambda_b_grid,
            nfold=self.nfold,
        )

        self.train_scores_ = compute_scores(
            self.fpca_eigenfunctions_,
            train_Y,
            mean_function=self.mean_function_,
        )

        (
            self.lmnn_,
            self.knn_,
            self.mahalanobis_matrix_,
            self.opt_neighbor_,
            self.opt_mu_,
            self.opt_K_,
            self.train_accuracy_,
            self.train_predictions_,
            self.cv_results_,
        ) = lmnn_grid(
            self.train_scores_,
            trainlabel,
            self.neighbors,
            self.mus,
            self.Ks,
            nfold=self.nfold,
        )

        self.classes_ = np.unique(trainlabel)
        return self

    def transform(self, Y):
        """Project functional observations onto the fitted FPCA basis.

        Parameters
        ----------
        Y : ndarray
            Functional data with shape ``(n_1, ..., n_D, N)``.

        Returns
        -------
        scores : ndarray
            FPC score matrix with shape ``(N, K)``.
        """
        self._check_is_fitted()
        Y = np.asarray(Y)
        self._validate_grid(Y)
        return compute_scores(
            self.fpca_eigenfunctions_,
            Y,
            mean_function=self.mean_function_,
        )

    def predict(self, Y):
        """Predict class labels for new functional observations."""
        scores = self.transform(Y)
        return self.knn_.predict(scores[:, :self.opt_K_])

    def score(self, Y, y):
        """Return mean classification accuracy on labeled functional data."""
        y = np.asarray(y)
        pred = self.predict(Y)
        if pred.shape[0] != y.shape[0]:
            raise ValueError("The number of labels must match the number of samples in Y.")
        return np.mean(pred == y)



## Optimal Rank-K marginal product basis (MPB)
def kompb(Y, Phis, Rlst, K, lambda_d1, lambda_d2, lambda_b,  maxiter = (100, 100), tol_inner = (1e-3, 1e-3), tol_outer = 1e-3, initialize = "random"):
    """
Fit a rank-K optimal marginal product basis (K-oMPB) representation
to multidimensional functional data using the MARGARITA algorithm.

Arguments:
    Y: ndarray
        Raw observed data tensor with shape (n_1, ..., n_D, N),
        where n_d is the number of observations along marginal
        dimension d and N is the number of subjects/samples.

    K: int
        Rank of the marginal product basis decomposition; i.e., the
        number of rank-1 marginal product basis functions to estimate.

    Phis: list of ndarrays
        Length-D list of marginal basis evaluation matrices.
        Phis[d] has shape (n_d, m_d), where m_d is the number of
        marginal basis functions used for dimension d.

    Rlst: list of ndarrays
        Length-D list of marginal roughness penalty matrices.
        Rlst[d] defines the roughness penalty associated with the
        marginal basis functions in dimension d. For example, these
        may be constructed from integrated squared second derivatives
        of the marginal basis functions.

    lambdas: array-like
        Length-(D+1) sequence of regularization parameters.

        The first D entries control the marginal roughness penalties:

            lambdas[d] = lambda_d,  d = 0, ..., D-1

        and the final entry controls regularization of the subject
        coefficient matrix:

            lambdas[-1] = lambda_b = lambda_{D+1}.

        Larger marginal lambda values impose stronger smoothness on
        the corresponding marginal basis functions. The effect of
        lambda_b depends on reg_type.

    reg_type: {"l1", "l2"}, default="l2"
        Type of regularization applied to the subject coefficient
        matrix. "l1" promotes sparse coefficients, while "l2"
        applies ridge regularization.

    maxiter: tuple of int, default=(200, 100)
        Maximum numbers of iterations for the MARGARITA optimization.
        The first entry is the maximum number of inner iterations
        used for the coefficient update, and the second entry is the
        maximum number of outer block coordinate descent iterations.

    tol_inner: tuple of float, default=(1e-3, 1e-3)
        Absolute and relative convergence tolerances, respectively,
        for the inner ADMM iterations used in the coefficient update.

    tol_outer: float, default=1e-3
        Convergence tolerance for the outer block coordinate descent
        iterations.

    initialize: {"svd", "random"}, default="random"
        Initialization method for the CP decomposition.

    verbose: bool, default=False
        If True, print optimization progress and convergence
        information.

Returns:
    Clst: list of ndarrays
        Length-D list of estimated marginal coefficient matrices.
        Clst[d] has shape (m_d, K), with column k containing the
        coefficients defining the kth marginal basis function in
        dimension d with respect to Phis[d].

    Smat: ndarray
        Subject coefficient matrix with shape (N, K). Each row
        contains the coefficients representing one subject in the
        estimated rank-K marginal product basis.

    scalars: float
        Scale factor associated with the normalized subject
        coefficient matrix returned by MARGARITA. The coefficients
        on their original fitted scale can be obtained as
        Smat * scalars.

    FLAG_C: bool
        Indicates whether the outer block coordinate descent
        algorithm satisfied its convergence tolerance.

    FLAG_N: bool
        Indicates whether the algorithm terminated because of
        evidence of numerical instability.

    itr: int
        Number of outer block coordinate descent iterations performed.
"""
    
    
    nmode = len(Y.shape[:-1]) 

    penalty_params = tuple([lambda_d1]*(nmode - 1) + [lambda_d2] + [lambda_b])

   # Perform SVDs basis evaluation matrices on Phi_d:Phi_d = UdDdV′d. 

    svdtuple = namedtuple("SVD", ["U", "s", "Vt"])

    Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]


    ## Perform the n-mode coordinate transformations into the spline coefficient space 
    ## Y_Bar should be close to zero so no need to center 
    G = tl.tenalg.multi_mode_dot(Y, [svdt.U.T for svdt in Svds], list(range(nmode)))
    Vs = [Svds[d].Vt.T for d in range(nmode)]
    Dinvs = [np.diag(1./Svds[d].s) for d in range(nmode)]
    Tlst_bcd = [Dinvs[d]@Vs[d].T@Rlst[d]@Vs[d]@Dinvs[d] for d in range(nmode)]

    # Ctilde: D rank-1 factors ctilde_d,k
						
    # ranks: 
                     
    Ctilde, Smat, scalars, FLAG_C, FLAG_N, itr = MARGARITA(G, Tlst_bcd, penalty_params, K, 
									 max_iter=maxiter, tol_inner=tol_inner, 
									 tol_outer=tol_outer,  regularization="l2", init=initialize, 
									verbose=False) # Ctilde: list of 3, (5,10) (5,10) (8,10)

	## Map factors to bpsline coordinate space
    Clst = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Ctilde[d] for d in range(nmode)] 
    Smat_scaled = np.multiply(Smat, scalars) #(50, 60)


    # ksi_dk = Phi_d*c_dk (K rank-1 marginal product functions )
    mpb_basis = np.zeros(tuple([K] + list(Y.shape[:-1])))
    
    for k in range(K):
      PhiCs = [Phis[d]@Clst[d][:, k] for d in range(nmode)] # (30, )
      result  = PhiCs[0]
      
# Perform tensor product using a loop
      for phicd in PhiCs[1:]:
        result = np.tensordot(result, phicd, 0) 
       
      mpb_basis[k,...] = result
      
    return mpb_basis, Clst, Smat_scaled, FLAG_C, FLAG_N, itr 




def tfold_splt(nfold, Nsamps):
    indices = np.arange(Nsamps)
    np.random.shuffle(indices)
    ho_indices = np.array_split(indices, nfold)
    tr_indices = []
    for e in ho_indices:
        mask_eval = np.ones(Nsamps, bool)
        mask_eval[e] = False
        tr_indices.append(indices[mask_eval])
    return ho_indices, tr_indices



#### cross-validation for selecting \lambda for K-oMPB' ####
def kompb_cross_val(Y, K, Phis, Rlst, lambda_d1_grid, lambda_d2_grid, lambda_b_grid, nfold=5, reg_type="l2", maxiter=(200, 100), 
					tol_inner=(1e-3, 1e-3), tol_outer=1e-3, initialize = "random", verbose=False):
    
    """
    K-oMPB with cross-validation to select penalty tuning paramters lambda
    
    Arguments:
    Y: ndarray
        Raw observed data tensor with shape
        (n_1, ..., n_D, N), where n_d is the number of observations
        along marginal dimension d and N is the number of subjects/samples.

    K: int
        Rank of the marginal product basis decomposition.

    Phis: list of ndarrays
        Length-D list of marginal basis evaluation matrices. Phis[d]
        has shape (n_d, m_d), where m_d is the number of basis
        functions used for marginal dimension d.

    Rlst: list of ndarrays
        Length-D list of marginal roughness penalty matrices.
        Rlst[d] is the penalty matrix associated with the basis functions
        for marginal dimension d.

    lambda_d1_grid: array-like
        Candidate values for the marginal roughness penalty lambda_d1.
        In the current implementation, lambda_d1 is applied to the first
        D-1 marginal dimensions. For D=3, it is applied to dimensions
        1 and 2.

    lambda_d2_grid: array-like
        Candidate values for the marginal roughness penalty lambda_d2.
        In the current implementation, lambda_d2 is applied to the last
        marginal dimension. For D=3, it is applied to dimension 3.

    lambda_b_grid: array-like
        Candidate values for the regularization parameter applied to the
        subject coefficient matrix B (S in MARGARITA). The form of the
        penalty is determined by reg_type.
   
    nfold: int, default=5
        Number of folds used for cross-validation.

    reg_type: {"l1", "l2"}, default="l2"
        Type of regularization applied to the subject coefficient matrix.
        "l1" applies an L1 penalty and "l2" applies an L2/ridge penalty.    
    
    maxiter: tuple of int, default=(200, 100)
        Maximum numbers of iterations for the MARGARITA optimization.
        The first entry is the maximum number of inner ADMM iterations,
        and the second entry is the maximum number of outer block
        coordinate descent iterations.

    tol_inner: tuple of float, default=(1e-3, 1e-3)
        Absolute and relative convergence tolerances, respectively, for
        the inner ADMM iterations used to update the coefficient matrix.

    tol_outer: float, default=1e-3
        Convergence tolerance for the outer block coordinate descent
        iterations in MARGARITA.

    initialize: {"svd", "random"}, default="random"
        Initialization method for the CP decomposition in MARGARITA.

    verbose: bool, default=False
        If True, print progress and cross-validation results for each
        combination of tuning parameters.

    Returns:
    cross_val_results: dict
        Dictionary mapping each tuning-parameter combination

            (lambda_d1, lambda_d2, lambda_b)

        to its mean validation sum of squared errors (SSE) across the
        nfold cross-validation folds. The parameter combination with the
        smallest value is the preferred combination according to the
        cross-validation criterion.    

    """
    nmode = len(Y.shape[:-1]) 
    ## Compute basis evaluation matrices and SVDs
    svdtuple = namedtuple("SVD", ["U", "s", "Vt"])
    Svds = [svdtuple(*np.linalg.svd(Phis[d], full_matrices=False)) for d in range(nmode)]
    ## Map to tilde space 
    Vs = [Svds[d].Vt.T for d in range(nmode)]
    Dinvs = [np.diag(1./Svds[d].s) for d in range(nmode)]
    Tlst_bcd = [Dinvs[d]@Vs[d].T@Rlst[d]@Vs[d]@Dinvs[d] for d in range(nmode)]
    ## BCD parameters     
    max_iter_inner, max_iter_outer = maxiter
    param_grid = [(lam_d1, lam_d2, lam_b) for lam_d1 in lambda_d1_grid for lam_d2 in lambda_d2_grid for lam_b in lambda_b_grid]
    ## construct folds 
    ho_indices, tr_indices = tfold_splt(nfold, Y.shape[-1])
    ## save cross-validation results 
    cross_val_results = {}
    for p_setting in range(len(param_grid)):
        lambda_d1, lambda_d2, lambda_b = param_grid[p_setting]
       # penalty_params = tuple([lambda_c]*nmode + [lambda_s])
        penalty_params = tuple([lambda_d1]*(nmode-1) + [lambda_d2] + [lambda_b])
        SSE_per_fold = np.zeros(nfold)
        for nf in range(nfold):
            Y_tr = Y[...,tr_indices[nf]]
            Y_ho = Y[...,ho_indices[nf]]
            G = tl.tenalg.multi_mode_dot(Y_tr, [svdt.U.T for svdt in Svds], list(range(nmode)))
            Ctilde, Smat, scalars, FLAG_C, FLAG_N, itr = tensor_decomposition.MARGARITA(G, Tlst_bcd, penalty_params, K, 
						                                         max_iter=maxiter, tol_inner=tol_inner, 
						                                         tol_outer=tol_outer,  regularization=reg_type, 
						                                        init=initialize, verbose=False)
            Clst = [Svds[d].Vt.T @ np.diag(1/Svds[d].s) @ Ctilde[d] for d in range(nmode)] 
            Smat_scaled = np.multiply(Smat, scalars)            
           # mpb_basis = np.zeros(tuple(list(Y.shape[:-1]) + [Smat_scaled.shape[1]]))
            mpb_basis = np.zeros(tuple(list(Y.shape[:-1]) + [Smat_scaled.shape[1]]))
            for k in range(Smat_scaled.shape[1]): ## write below to be a general outer product
        
               PhiCs = [Phis[d]@Clst[d][:, k] for d in range(nmode)] # (30, )
               result  = PhiCs[0]
# Perform tensor product using a loop
               for phicd in PhiCs[1:]:
                  result = np.tensordot(result, phicd, 0) 
       
               mpb_basis[...,k] = result
      
            ## compute MSE on validation set 
            Y_ho_D1 = tl.unfold(Y_ho, nmode).T
            Z_D1 = tl.unfold(mpb_basis, nmode).T
            Gram_matrix_Zeta = Z_D1.T @ Z_D1
            S_init = np.zeros((Y_ho.shape[-1], Smat_scaled.shape[1])) ##could give this a "warm-start" with a simple linear regression
            U_dual_init = np.zeros(S_init.shape)
            S_ho, _ = tensor_decomposition.update_S(Y_ho_D1, Z_D1, Gram_matrix_Zeta, S_init, U_dual_init, lambda_b, tol_inner, max_iter_inner, reg_type)
            Y_ho_hat = tl.tenalg.mode_dot(mpb_basis, S_ho, nmode)
            SSE = tl.norm(Y_ho - Y_ho_hat)**2

            SSE_per_fold[nf] = SSE

        cross_val_results[(lambda_d1, lambda_d2, lambda_b)] = np.mean(SSE_per_fold) 

        if verbose:
            print("Finished CV for lambda_c=%s, lambda_s=%s. Value: %s"%(lambda_d1, lambda_d2, lambda_b, cross_val_results[(lambda_d1, lambda_d2, lambda_b)]))

    return cross_val_results




## Wrapper function for two stage FPCA from [1]

def two_stage_fpca(mpb_basis, bspline_basis, Clst, Smat_scaled):
    """
    Implementation of roughness penalized FPCA from Silverman 1996, adapted to marginal product basis functions 
    Arguments:
        mpb_basis: ksi_dk = Phi_d*c_dk (K rank-1 marginal product functions)
        bspline_basis: specified basis functions
        Clst: \hat{C}_d
        Smat_scaled: (ndarray) N x K matrix of subject coefficients w.r.t. the marginal product basis functions. B matrix in the paper
    
    Returns:  
        fpca_eigenfunctions: eigenfunctions (K, n1, n2, n3)
        B: KxK matrix of coefficient expansion of the eigenfunctions (s_j in the paper)
        fpca_eigenvalues: (K,) array of associated eigenvalues 
    
    """
    mpb = MPB(bspline_basis, Clst)
    ## Perform FPCA 
    J = mpb.gram_matrix()
    R = mpb.roughness_matrix()
    B, fpca_eigenvalues = FPCA(Smat_scaled, J, R, lam=1e-10)
    fpca_eigenfunctions = tl.tenalg.mode_dot(mpb_basis, B.T, 0)
    
    return fpca_eigenfunctions, B, fpca_eigenvalues



def fpca_cv(Y, K, bspline_basis, xgrids, nmode,
            lambda_d1_grid, lambda_d2_grid, lambda_b_grid, nfold=5):
    """
    Estimate a K-oMPB representation, select regularization parameters
    by cross-validation, and perform two-stage penalized FPCA.

    The function first constructs the marginal B-spline basis evaluation
    matrices and second-order derivative roughness penalty matrices.
    It then uses cross-validation to select the marginal roughness
    parameters and subject-coefficient regularization parameter for
    MARGARITA. The model is refit using the selected parameters, followed
    by the two-stage FPCA procedure. The resulting eigenfunctions are
    normalized and their variance explained and approximation to a
    reference set of eigenfunctions are calculated.

    Arguments:
        Y: ndarray
            Raw observed multidimensional functional data tensor with
            shape (n_1, ..., n_D, N), where n_d is the number of
            observations along marginal dimension d and N is the number
            of subjects/samples.

        K: int
            Rank of the marginal product basis (K-oMPB), i.e. the number
            of rank-1 marginal product basis functions estimated by
            MARGARITA.

        bspline_basis: list
            Length-D list of B-spline basis objects, one for each
            marginal dimension. Each basis object is used to evaluate
            the marginal basis functions and to construct their inner
            product and roughness penalty matrices.

        xgrids: list of ndarray
            Length-D list containing the grid points for each marginal
            domain. xgrids[d] is used to evaluate the B-spline basis for
            dimension d.

        nmode: int
            Number of functional dimensions D.

        lambda_d1_grid: array-like
            Candidate values for the marginal roughness parameter
            lambda_d1. In the current implementation, this parameter is
            applied to the first D-1 marginal dimensions.

            For D=3:
                lambda_1 = lambda_2 = lambda_d1.

        lambda_d2_grid: array-like
            Candidate values for the marginal roughness parameter
            lambda_d2. In the current implementation, this parameter is
            applied to the final marginal dimension.

            For D=3:
                lambda_3 = lambda_d2.

        lambda_b_grid: array-like
            Candidate values for the regularization parameter applied to
            the subject coefficient matrix B (or S in MARGARITA).
            In this function, an L2/ridge penalty is used.

        Psi_tensor: ndarray
            Reference eigenfunction tensor used to assess the agreement
            between the estimated eigenfunctions and a reference set.
            The first dimension indexes the eigenfunctions. The tensor
            should have the same spatial dimensions as fpca_eigenfunctions for the
            eigenfunctions being compared.

    Returns:
        fpca_eigenfunctions: ndarray
            Estimated and normalized eigenfunction tensor obtained from
            the two-stage FPCA procedure. The first dimension indexes
            eigenfunctions and the remaining dimensions correspond to
            the functional domain.

        mpb_basis: ndarray
            Estimated K-oMPB basis functions evaluated on the original
            observation grid. These are the fitted marginal product basis
            functions produced by MARGARITA.

        B: ndarray
            Matrix of subject/eigenfunction coefficients produced by the
            two-stage FPCA procedure.

        fpca_eigenvalues: ndarray
            Estimated eigenvalues or variance contributions associated
            with the estimated eigenfunctions.

        PVE: ndarray
            Cumulative proportion of variance explained by the estimated
            eigenfunctions, computed as

                PVE[k] = sum_{j=1}^{k} fpca_eigenvalues[j] / sum_j fpca_eigenvalues[j].

        lam_d1: float
            Selected value of lambda_d1 from the cross-validation grid.

        lam_d2: float
            Selected value of lambda_d2 from the cross-validation grid.

        lam_b: float
            Selected value of lambda_b from the cross-validation grid.

        FLAG_C: bool
            Indicates whether the MARGARITA block coordinate descent
            algorithm satisfied its convergence criterion when refit on
            the full dataset.

        FLAG_N: bool
            Indicates whether MARGARITA detected numerical instability
            when refit on the full dataset.
    """
    
    # Compute basis evaluation matrices Phi_d, d = 1, ..., D
    Phis = [np.squeeze(bspline_basis[d].evaluate(xgrids[d])).T for d in range(nmode)]

## Specify differential operator for penalization 
    D2 = LinearDifferentialOperator(2)
    Rlst = [gram_matrix(D2, bspline_basis[d]) for d in range(nmode)] 


    cross_val_results = kompb_cross_val(Y, K, Phis, Rlst, lambda_d1_grid, lambda_d2_grid, lambda_b_grid, nfold=nfold, reg_type="l2", maxiter=(500, 500), 
					tol_inner=(1e-3, 1e-3), tol_outer=1e-4, initialize = "random", verbose=False)    


    cross_val_results_tuple = [(k,v) for k, v in cross_val_results.items()]
    cross_val_selection = sorted(cross_val_results_tuple, key=lambda e: e[1], reverse=False)[0]
    lam_d1, lam_d2, lam_b = cross_val_selection[0]     

# refit on full data with the optimal lambdas

    mpb_basis, Clst, Smat_scaled, FLAG_C, FLAG_N, itr  = kompb(Y, Phis, Rlst, K, lambda_d1 = lam_d1, lambda_d2 = lam_d2, lambda_b= lam_b,  maxiter = (500, 500), tol_inner = (1e-3, 1e-3), tol_outer = 1e-4, initialize = "random")


    fpca_eigenfunctions, B, fpca_eigenvalues = two_stage_fpca(mpb_basis, bspline_basis, Clst, Smat_scaled)
    
    ## normalize fpca_eigenfunctions
    norm_fpca_eigenfunctions = fpca_eigenfunctions
    n_grid = np.prod(fpca_eigenfunctions.shape[1:])
    
    for k in range(fpca_eigenfunctions.shape[0]):
        
        norm_fpca_eigenfunctions[k, ...] = fpca_eigenfunctions[k, ...] / np.sqrt(
            tl.tenalg.inner(fpca_eigenfunctions[k, ...], fpca_eigenfunctions[k, ...]) / n_grid
        )
        
        
    fpca_eigenfunctions = norm_fpca_eigenfunctions    

    PVE = np.cumsum(fpca_eigenvalues)/np.sum(fpca_eigenvalues)
   


    return fpca_eigenfunctions, mpb_basis, B, fpca_eigenvalues, PVE, lam_d1, lam_d2, lam_b, FLAG_C, FLAG_N



def compute_scores(fpca_eigenfunctions, Y, mean_function=None):
    """Compute FPC scores by least-squares projection.

    Parameters
    ----------
    fpca_eigenfunctions : ndarray
        Estimated eigenfunction tensor with shape ``(K, n_1, ..., n_D)``.
    Y : ndarray
        Functional observations with shape ``(n_1, ..., n_D, N)``.
    mean_function : ndarray or None, default=None
        Mean function used to center ``Y``. It should have shape
        ``(n_1, ..., n_D, 1)``. If ``None``, the sample mean of ``Y`` is
        used. For prediction after fitting FLMNN, pass the training mean.

    Returns
    -------
    scores : ndarray
        FPC score matrix with shape ``(N, K)``.
    """
    fpca_eigenfunctions = np.asarray(fpca_eigenfunctions)
    Y = np.asarray(Y)

    K = fpca_eigenfunctions.shape[0]
    Nsample = Y.shape[-1]

    if tuple(fpca_eigenfunctions.shape[1:]) != tuple(Y.shape[:-1]):
        raise ValueError(
            "fpca_eigenfunctions and Y must have matching functional grid dimensions."
        )

    fpca_eigenfunctions_flat = fpca_eigenfunctions.reshape(K, -1)
    F = fpca_eigenfunctions_flat @ fpca_eigenfunctions_flat.T

    if mean_function is None:
        mean_function = Y.mean(axis=-1, keepdims=True)
    else:
        mean_function = np.asarray(mean_function)
        expected_shape = Y.shape[:-1] + (1,)
        if mean_function.shape != expected_shape:
            raise ValueError(
                f"mean_function must have shape {expected_shape}, got {mean_function.shape}."
            )

    Y_centered = Y - mean_function
    Yall_flat = Y_centered.reshape(-1, Nsample)
    Y_tilde = fpca_eigenfunctions_flat @ Yall_flat
    scores = (np.linalg.pinv(F) @ Y_tilde).T
    return scores


from metric_learn import LMNN
from sklearn.neighbors import KNeighborsClassifier


def lmnn_grid(train_scores, trainlabel, neighbors, mus, Ks, nfold=5):
    """Select LMNN/KNN tuning parameters using training data only.

    Parameters
    ----------
    train_scores : ndarray
        Training FPC score matrix with shape ``(N_train, K_max)``.
    trainlabel : array-like
        Training class labels with shape ``(N_train,)``.
    neighbors : array-like of int
        Candidate numbers of target/KNN neighbors.
    mus : array-like of float
        Candidate LMNN regularization parameters.
    Ks : array-like of int
        Candidate numbers of FPC score components.
    nfold : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    lmnn : LMNN
        Final LMNN model fitted on all training observations.
    knn : KNeighborsClassifier
        Final KNN classifier using the learned LMNN metric.
    mahalanobis_matrix : ndarray
        Learned Mahalanobis matrix with shape ``(opt_K, opt_K)``.
    opt_neighbor : int
        Selected number of neighbors.
    opt_mu : float
        Selected LMNN regularization parameter.
    opt_K : int
        Selected number of FPC score components.
    trainacc : float
        Final training classification accuracy.
    trainpred : ndarray
        Final training predictions.
    cross_val_results : dict
        Mapping ``(neighbor, mu, K)`` to mean validation accuracy.
    """
    train_scores = np.asarray(train_scores)
    trainlabel = np.asarray(trainlabel)

    if train_scores.shape[0] != trainlabel.shape[0]:
        raise ValueError("train_scores and trainlabel must contain the same number of samples.")
    if nfold < 2:
        raise ValueError("nfold must be at least 2.")

    param_grid = [
        (neighbor, mu, K)
        for neighbor in neighbors
        for mu in mus
        for K in Ks
    ]
    if not param_grid:
        raise ValueError("The LMNN parameter grid must not be empty.")

    max_components = train_scores.shape[1]
    for neighbor, _, K in param_grid:
        if K < 1 or K > max_components:
            raise ValueError(
                f"Each K must satisfy 1 <= K <= {max_components}; received {K}."
            )
        if neighbor < 1:
            raise ValueError("Neighbor counts must be positive integers.")

    ho_indices, tr_indices = tfold_splt(nfold, train_scores.shape[0])
    cross_val_results = {}

    for neighbor, mu, K in param_grid:
        ACC_per_fold = np.zeros(nfold)

        for nf in range(nfold):
            if neighbor >= len(tr_indices[nf]):
                raise ValueError(
                    f"neighbor={neighbor} must be smaller than the number of "
                    f"training observations in each CV fold ({len(tr_indices[nf])})."
                )

            lmnn = LMNN(n_neighbors=neighbor, regularization=mu)
            lmnn.fit(
                train_scores[tr_indices[nf], :K],
                trainlabel[tr_indices[nf]],
            )

            knn = KNeighborsClassifier(
                n_neighbors=neighbor,
                metric=lmnn.get_metric(),
            )
            knn.fit(
                train_scores[tr_indices[nf], :K],
                trainlabel[tr_indices[nf]],
            )

            valpred = knn.predict(train_scores[ho_indices[nf], :K])
            ACC_per_fold[nf] = np.mean(valpred == trainlabel[ho_indices[nf]])

        cross_val_results[(neighbor, mu, K)] = np.mean(ACC_per_fold)

    opt_neighbor, opt_mu, opt_K = max(
        cross_val_results,
        key=cross_val_results.get,
    )

    lmnn = LMNN(n_neighbors=opt_neighbor, regularization=opt_mu)
    lmnn.fit(train_scores[:, :opt_K], trainlabel)
    mahalanobis_matrix = lmnn.get_mahalanobis_matrix()

    knn = KNeighborsClassifier(
        n_neighbors=opt_neighbor,
        metric=lmnn.get_metric(),
    )
    knn.fit(train_scores[:, :opt_K], trainlabel)

    trainpred = knn.predict(train_scores[:, :opt_K])
    trainacc = np.mean(trainlabel == trainpred)

    return (
        lmnn,
        knn,
        mahalanobis_matrix,
        opt_neighbor,
        opt_mu,
        opt_K,
        trainacc,
        trainpred,
        cross_val_results,
    )

