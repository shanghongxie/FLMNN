#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 10:21:01 2026

@author: shanghongxie
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_eigenfunction_slice(
    estimated_eigenfunctions,
    k=0,
    third_idx=None,
    true_eigenfunctions=None,
):
    """
    Plot a two-dimensional slice of an estimated eigenfunction.

    If true eigenfunctions are provided, the true and estimated
    eigenfunctions are shown side by side for comparison.

    Parameters
    ----------
    estimated_eigenfunctions : ndarray
        Estimated eigenfunctions with shape

            (n_components, n1, n2, n3).

    k : int, default=0
        Index of the eigenfunction to plot.

    third_idx : int, optional
        Grid index at which the third functional dimension is fixed.
        If None, the midpoint of the third dimension is used.

    true_eigenfunctions : ndarray, optional
        True eigenfunctions with shape

            (n_components, n1, n2, n3).

        This is typically available only in simulation studies.
        If provided, the true and estimated eigenfunctions are
        displayed side by side. The sign of the estimated
        eigenfunction is aligned with the true eigenfunction before
        plotting.

    Returns
    -------
    None
        Displays the eigenfunction slice.
    """

    estimated_eigenfunctions = np.asarray(estimated_eigenfunctions)


    if estimated_eigenfunctions.ndim != 4:
        raise ValueError(
            "estimated_eigenfunctions must have shape "
            "(n_components, n1, n2, n3)."
        )

    if k < 0 or k >= estimated_eigenfunctions.shape[0]:
        raise ValueError(
            f"k must be between 0 and "
            f"{estimated_eigenfunctions.shape[0] - 1}."
        )


    if third_idx is None:
        third_idx = estimated_eigenfunctions.shape[3] // 2

    if third_idx < 0 or third_idx >= estimated_eigenfunctions.shape[3]:
        raise ValueError(
            f"third_idx must be between 0 and "
            f"{estimated_eigenfunctions.shape[3] - 1}."
        )

    # ------------------------------------------------------------
    # Estimated eigenfunction
    # ------------------------------------------------------------
    estimated_k = estimated_eigenfunctions[k].copy()

    # ------------------------------------------------------------
    # Grid for dimensions 1 and 2
    # ------------------------------------------------------------
    n1 = estimated_k.shape[0]
    n2 = estimated_k.shape[1]

    x1 = np.linspace(0, 1, n1)
    x2 = np.linspace(0, 1, n2)

    X1, X2 = np.meshgrid(x1, x2, indexing="ij")

    # ------------------------------------------------------------
    # true eigenfunctions are NOT available
    # ------------------------------------------------------------
    if true_eigenfunctions is None:

        estimated_slice = estimated_k[:, :, third_idx]

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            X1,
            X2,
            estimated_slice
        )

        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_zlabel("Eigenfunction value")

        ax.set_title(
            f"Estimated eigenfunction {k + 1}\n"
            f"Third-dimension index = {third_idx}"
        )

        plt.tight_layout()
        plt.show()

        return

    # ------------------------------------------------------------
    # true eigenfunctions are available
    # ------------------------------------------------------------
    true_eigenfunctions = np.asarray(true_eigenfunctions)

    if true_eigenfunctions.ndim != 4:
        raise ValueError(
            "true_eigenfunctions must have shape "
            "(n_components, n1, n2, n3)."
        )

    if true_eigenfunctions.shape[1:] != estimated_eigenfunctions.shape[1:]:
        raise ValueError(
            "True and estimated eigenfunctions must be evaluated "
            "on the same functional grid."
        )

    if k >= true_eigenfunctions.shape[0]:
        raise ValueError(
            f"true_eigenfunctions contains only "
            f"{true_eigenfunctions.shape[0]} components."
        )

    true_k = true_eigenfunctions[k].copy()

    # ------------------------------------------------------------
    # Align eigenfunction signs
    #
    # psi and -psi represent the same eigenfunction.
    # ------------------------------------------------------------
    inner_product = np.sum(true_k * estimated_k)

    if inner_product < 0:
        estimated_k *= -1

    # Normalized similarity over the ENTIRE 3D eigenfunction
    similarity = np.abs(
        np.sum(true_k * estimated_k)
        /
        (
            np.linalg.norm(true_k)
            * np.linalg.norm(estimated_k)
        )
    )

    # Extract the two-dimensional slices
    true_slice = true_k[:, :, third_idx]
    estimated_slice = estimated_k[:, :, third_idx]

    # Use the same vertical scale
    zmin = min(
        true_slice.min(),
        estimated_slice.min()
    )

    zmax = max(
        true_slice.max(),
        estimated_slice.max()
    )

    # ------------------------------------------------------------
    # Plot true and estimated eigenfunctions side by side
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection="3d")

    ax1.plot_surface(
        X1,
        X2,
        true_slice
    )

    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.set_zlabel("Eigenfunction value")
    ax1.set_zlim(zmin, zmax)
    ax1.set_title(f"True eigenfunction {k + 1}")

    ax2 = fig.add_subplot(122, projection="3d")

    ax2.plot_surface(
        X1,
        X2,
        estimated_slice
    )

    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    ax2.set_zlabel("Eigenfunction value")
    ax2.set_zlim(zmin, zmax)
    ax2.set_title(f"Estimated eigenfunction {k + 1}")

    fig.suptitle(
        f"Third-dimension index = {third_idx} | "
        f"Similarity = {similarity:.3f}"
    )

    plt.tight_layout()
    plt.show()