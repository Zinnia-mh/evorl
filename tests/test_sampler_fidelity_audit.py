"""Exact probability-audit arithmetic must reproduce the actual search path."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scripts.validate_replay_sampling_fidelity import exact_binary_counts


@pytest.mark.parametrize("bits", [8, 13])
@pytest.mark.parametrize("cdf", [
    [0., .1, .2, .2, .6, .9, 1., 1.],
    [0., .2, .1, .6, .59, .9, 1., 1.],
])
def test_binary_partition_matches_full_uniform_grid(cdf, bits):
    array = np.asarray(cdf, dtype=np.float32)
    grid = np.arange(2**bits, dtype=np.float32) / np.float32(2**bits)
    indices = np.asarray(jnp.searchsorted(jnp.asarray(array), jnp.asarray(grid), side="right"))
    expected = np.bincount(np.minimum(indices, len(array)-1), minlength=len(array))
    np.testing.assert_array_equal(exact_binary_counts(array, bits), expected)


def test_binary_partition_resolves_float64_52_bit_boundaries():
    scale = 2**52
    boundaries = np.array(
        [0, 1, 2**26 + 1, 2**51 - 1, 2**51, scale - 1, scale, scale],
        dtype=np.int64,
    )
    cdf = boundaries.astype(np.float64) / scale
    np.testing.assert_array_equal(
        exact_binary_counts(cdf, 52), np.diff(np.r_[0, boundaries])
    )
    probes = np.unique(np.r_[boundaries, boundaries - 1])
    probes = probes[(probes >= 0) & (probes < scale)]
    with jax.enable_x64(True):
        actual = np.asarray(jnp.searchsorted(
            jnp.asarray(cdf), jnp.asarray(probes / scale), side="right"
        ))
    np.testing.assert_array_equal(
        actual, np.searchsorted(boundaries, probes, side="right")
    )
