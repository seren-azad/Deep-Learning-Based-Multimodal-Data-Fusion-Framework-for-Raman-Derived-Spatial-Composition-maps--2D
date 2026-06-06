from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, chisquare
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def pixel_ks_test(images_a: np.ndarray, images_b: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov test for two pixel-intensity distributions."""

    return ks_2samp(images_a.ravel(), images_b.ravel())


def pixel_chi_square_test(images_a: np.ndarray, images_b: np.ndarray, bins: int = 10) -> tuple[float, float]:
    edges = np.linspace(0, 1, bins + 1)
    hist_a, _ = np.histogram(images_a.ravel(), bins=edges)
    hist_b, _ = np.histogram(images_b.ravel(), bins=edges)
    hist_b = np.where(hist_b == 0, 1, hist_b)
    return chisquare(f_obs=hist_a, f_exp=hist_b)


def pca_projection(array: np.ndarray, n_components: int = 2, random_state: int = 42):
    flat = array.reshape(array.shape[0], -1)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(flat), pca


def random_domain_split(images: np.ndarray, test_size: float = 0.5, random_state: int = 42):
    indices = np.arange(images.shape[0])
    idx_a, idx_b = train_test_split(indices, test_size=test_size, random_state=random_state)
    return images[idx_a], images[idx_b]
