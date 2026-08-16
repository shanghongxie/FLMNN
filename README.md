# FLMNN Python Library

**Functional Large Margin Nearest Neighbor (FLMNN)** is a method for classification of multidimensional functional data. 

- Authors: **Shanghong Xie<sup>a,b</sup> (sx2@mailbox.sc.edu), and R. Todd Ogden<sup>b</sup>**

- Affiliations:
   + 1. **Department of Statistics, University of South Carolina, Columbia, SC, USA**
   + 2. **Department of Biostatistics, Mailman School of Public Health, Columbia University, New York, NY, USA**

The main user-facing interface follows a scikit-learn-style workflow:

```python
from flmnn import FLMNN

model = FLMNN(...)
model.fit(train_Y, trainlabel)
testpred = model.predict(test_Y)
```

## Project files

```text
flmnn.py
    Main FLMNN estimator and supporting model-fitting routines.

genData.py
    Simulation-data generators for dense and sparse/irregular
    multidimensional functional data, plus sparse-data interpolation.

plotting.py
    Visualization utilities, including eigenfunction-slice plots.

example.py
    Short example showing the basic dense data FLMNN workflow.

sim_example.py
    Extended simulation examples for dense and sparse/irregular
    functional predictors.
```

## Installation

Create a dedicated Python environment for FLMNN. The implementation has been developed with the following main packages:

- NumPy
- SciPy
- scikit-fda
- TensorLy
- scikit-learn
- metric-learn
- matplotlib

For the current implementation, use **scikit-fda 0.10.1**.

A typical environment setup is:

```bash
conda create -n flmnn python=3.12
conda activate flmnn
python -m pip install numpy scipy pandas scikit-fda==0.10.1 tensorly scikit-learn metric-learn matplotlib
```

`metric-learn` and `scikit-learn` should be installed at mutually compatible versions. If `metric-learn` raises an error involving `force_all_finite`, use a scikit-learn version that still supports that argument.

## Data format

FLMNN expects functional observations in an `ndarray` whose **last axis indexes subjects**.

For a three-dimensional functional domain:

```text
Y.shape = (n1, n2, n3, N)
```

where `n1`, `n2`, and `n3` are the numbers of grid points along the three functional dimensions, and `N` is the number of subjects.

For example:

```python
train_Y.shape
# (10, 10, 64, 100)
```

means that 100 subjects are observed on a `10 x 10 x 64` functional grid.

## Quick start

The following is the basic dense data example used by `example.py`.

### 1. Generate simulated functional data

```python
from genData import generate_dense_data


data = generate_dense_data(
    Ns=(4, 4, 16),
    Ms=(4, 4, 4),
    npc=10,
    Ntrain=100,
    Ntest=1000,
    sigma2x=1,
    sigma2y=1,
    random_state=1,
)
```

`generate_dense_data()` constructs tensor-product B-spline eigenfunctions, generates exponentially decaying eigenvalues, FPC scores, constructs the functional observations, adds measurement noise, and generates class labels from a noisy transformation of the true FPC scores. It returns both training and test data.

The returned dictionary includes, among other quantities:

```python
train_Y = data["Y_noisy"]
trainlabel = data["trainlabel"]

test_Y = data["test_Y_noisy"]
testlabel = data["testlabel"]

true_eigenfunctions = data["true_eigenfunctions"]
eigenvalues = data["eigenvalues"]
train_true_scores = data["train_true_scores"]
test_true_scores = data["test_true_scores"]
```

### 2. Define the FLMNN model

```python
from flmnn import FLMNN

model = FLMNN(
    Ms=(4, 4, 4),
    K=30,
    lambda_d1_grid=(1e-11, 1e-10),
    lambda_d2_grid=(1e-11, 1e-10),
    lambda_b_grid=(1e-8,),
    neighbors=(5,),
    mus=(0.5,),
    Ks=(10,),
)
```

### 3. Fit on labeled training data

```python
model.fit(train_Y, trainlabel)
```

### 4. Predict new subjects

```python
testpred = model.predict(test_Y)
```

The test data are projected onto the FPCA representation learned from the training data and then classified using the fitted LMNN model.

### 5. Evaluate classification accuracy

```python
import numpy as np

testacc = np.mean(testpred == testlabel)
print("Test accuracy:", testacc)
```

or use the estimator's `score()` method:

```python
testacc = model.score(test_Y, testlabel)
print("Test accuracy:", testacc)
```

## FLMNN parameters

### `Ms`

Number of marginal B-spline basis functions for each functional dimension.

For example:

```python
Ms=(5, 5, 8)
```

uses 5, 5, and 8 B-spline basis functions in the three dimensions.

### `K`

Rank of the optimal marginal product basis used by MARGARITA. The fitted representation contains `K` rank-one marginal product basis functions.

### `lambda_d1_grid`

Candidate marginal roughness penalties for the first `D-1` functional dimensions. For the current three-dimensional implementation,

```text
lambda_1 = lambda_2 = lambda_d1
```

### `lambda_d2_grid`

Candidate marginal roughness penalties for the final functional dimension. For three-dimensional data,

```text
lambda_3 = lambda_d2
```

### `lambda_b_grid`

Candidate regularization parameters for the subject coefficient matrix in MARGARITA.

### `neighbors`

Candidate numbers of nearest neighbors used in LMNN and the final KNN classifier.

### `mus`

Candidate LMNN regularization parameters.

### `Ks`

Candidate numbers of FPC scores used for metric learning and classification.


### `nfold`

Number of cross-validation folds. The default is 5.

### `basis_order`

Order of the marginal B-spline basis. The default is 4.

## Fitted model outputs

After fitting, important quantities are stored as model attributes.

### Estimated FPCA eigenfunctions

```python
estimated_eigenfunctions = model.fpca_eigenfunctions_
```

The first axis indexes the estimated eigenfunctions, with the remaining axes corresponding to the functional domain.

For example:

```python
estimated_eigenfunctions[0, ...]
```

is the first estimated functional eigenfunction.

### Estimated FPCA eigenvalues

```python
estimated_eigenvalues = model.fpca_eigenvalues_
```

These quantify the estimated variation associated with the functional principal components.

### Proportion of variance explained

```python
pve = model.pve_
```

`pve` is the cumulative proportion of variance explained by the estimated FPCA eigenfunctions.

### Estimated marginal product basis

```python
mpb_basis = model.mpb_basis_
```

These are the fitted K-oMPB functions obtained from MARGARITA.

### Selected smoothing parameters

```python
model.lambda_d1_
model.lambda_d2_
model.lambda_b_
```

These are the selected values from the MARGARITA cross-validation step.

### Selected LMNN/KNN parameters

```python
model.opt_neighbor_
model.opt_mu_
model.opt_K_
```

These are the selected number of neighbors, LMNN regularization parameter, and number of FPC scores used for classification.

### Learned Mahalanobis matrix

```python
M = model.mahalanobis_matrix_
```

### FPC scores

Training scores learned during fitting are available as:

```python
train_scores = model.train_scores_
```

Scores for new observations can be obtained without prediction using:

```python
new_scores = model.transform(new_Y)
```

## Simulation: dense and irregular functional data

`sim_example.py` contains code for simulation studies. Scenario 1 (dense and irregular functional data) uses:

```python
data = generate_dense_data(
    Ns=(10, 10, 64),
    Ms=(5, 5, 8),
    npc=10,
    Ntrain=100,
    Ntest=1000,
    sigma2x=1,
    sigma2y=5,
    random_state=1,
)
```

The example then fits:

```python
model = FLMNN(
    Ms=(5, 5, 8),
    K=30,
    lambda_d1_grid=(1e-11, 1e-10, 1e-8),
    lambda_d2_grid=(1e-11, 1e-10, 1e-8),
    lambda_b_grid=(1e-10, 1e-8, 1e-6),
    neighbors=(5,),
    mus=(0.5,),
    Ks=(10,),
)

model.fit(train_Y, trainlabel)
testpred = model.predict(test_Y)
```

The example also extracts the learned Mahalanobis matrix, selected smoothing parameters, estimated eigenfunctions, and estimated eigenvalues.

## Simulation: sparse and irregular functional data

`genData.py` also provides `generate_sparse_data()` for sparse/irregular functional predictors.

Example settings from `sim_example.py` are:

```python
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
```

The generator returns separate sparse training and test observations:

```python
train_sparse_Y = data["train_sparse_Y"]
test_sparse_Y = data["test_sparse_Y"]
```

Interpolation is available for sparse and irregular data:

```python
from genData import interpolate_sparse_data

train_Y_interp = interpolate_sparse_data(
    train_sparse_Y,
    data["train_mask"],
)

test_Y_interp = interpolate_sparse_data(
    test_sparse_Y,
    data["test_mask"],
)
```

The interpolated data can then be supplied to FLMNN:

```python
model.fit(train_Y_interp, trainlabel)
testpred = model.predict(test_Y_interp)
```

For the current sparse data design, `n_locations` specifies the number of observed locations in the first two dimensions per subject. The complete trajectory along the third dimension is retained at each selected location.

## Eigenfunction visualization

`plotting.py` contains utilities for visualizing estimated eigenfunctions.

For real data, where true eigenfunctions are unknown:

```python
from plotting import plot_eigenfunction_slice

plot_eigenfunction_slice(
    model.fpca_eigenfunctions_,
    k=0,
    third_idx=10,
)
```

For simulation studies, the true and estimated eigenfunctions can be compared directly:

```python
plot_eigenfunction_slice(
    model.fpca_eigenfunctions_,
    k=0,
    third_idx=10,
    true_eigenfunctions=true_eigenfunctions,
)
```

Here `k=0` selects the first eigenfunction and `third_idx=10` fixes the third functional dimension at a selected grid index.

## Evaluating eigenfunction recovery

Because the eigenfunctions are known in simulation, their recovery can be evaluated quantitatively.

A useful similarity measure is the absolute normalized inner product:
$\frac{|\langle \psi_k, \hat{\psi}_k \rangle|}
{\|\psi_k\|\|\hat{\psi}_k\|}$,

Values close to 1 indicate strong agreement. The absolute value accounts for the fact that an eigenfunction and its negative represent the same eigenfunction.

The simulation also provides true FPC scores and eigenvalues, so score and eigenvalue recovery can be evaluated separately from classification accuracy.

## Examples

### Simple example

Run:

```bash
python example.py
```

This example demonstrates the basic dense data workflow using `generate_dense_data()`, `FLMNN.fit()`, `FLMNN.predict()`, and test set accuracy.

### Full simulation example

Run:

```bash
python sim_example.py
```

This example demonstrates both:

1. dense and regular functional predictors; and
2. sparse and irregular functional predictors with interpolation.

in the simulation studies.
