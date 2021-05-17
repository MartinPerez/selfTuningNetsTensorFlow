from typing import Iterator

import numpy as np
import tensorflow as tf


def x_interval_batches(
    iter_seed: int,
    max_batches: int,
    batch_size: int,
    x_bound: float = 1.0
) -> Iterator[np.ndarray]:
    """Yields up to max_batches of (-x_bound, x_bound)"""
    rng = np.random.default_rng(iter_seed)
    for _ in range(max_batches):
        yield rng.uniform(-x_bound, x_bound, batch_size).astype(np.float32)


def standard_normal_batches(
    iter_seed: int,
    max_batches: int,
    batch_size: int
) -> Iterator[np.ndarray]:
    """Yields up to max_batches of standard normal"""
    rng = np.random.default_rng(iter_seed)
    for _ in range(max_batches):
        yield rng.normal(size=batch_size).astype(np.float32)
