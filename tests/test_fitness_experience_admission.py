import jax.numpy as jnp
import numpy as np

from scripts.fitness_experience_admission import (
    select_admitted_individuals,
    transition_admission_mask,
)


def test_fitness_top_selects_exact_top16():
    fitnesses = jnp.asarray(
        [13, 2, 27, 1, 31, 18, 6, 22, 15, 5, 28, 17, 11, 30, 9, 21,
         12, 4, 25, 8, 24, 0, 19, 29, 7, 16, 10, 26, 3, 23, 14, 20],
        dtype=jnp.float32,
    )
    selected = np.asarray(select_admitted_individuals(
        fitnesses,
        mode="fitness_top",
        admission_count=16,
        seed=3083,
        generation=11,
    ))
    expected = np.sort(np.argsort(-np.asarray(fitnesses), kind="stable")[:16])
    np.testing.assert_array_equal(selected, expected)
    rejected = np.setdiff1d(np.arange(32), selected)
    assert np.min(np.asarray(fitnesses)[selected]) >= np.max(
        np.asarray(fitnesses)[rejected]
    )


def test_random16_is_reproducible_and_fitness_independent():
    first = np.asarray(select_admitted_individuals(
        jnp.arange(32, dtype=jnp.float32),
        mode="random",
        admission_count=16,
        seed=3079,
        generation=37,
    ))
    second = np.asarray(select_admitted_individuals(
        -jnp.arange(32, dtype=jnp.float32),
        mode="random",
        admission_count=16,
        seed=3079,
        generation=37,
    ))
    next_generation = np.asarray(select_admitted_individuals(
        jnp.arange(32, dtype=jnp.float32),
        mode="random",
        admission_count=16,
        seed=3079,
        generation=38,
    ))
    np.testing.assert_array_equal(first, second)
    assert len(first) == 16
    assert len(np.unique(first)) == 16
    assert np.all((0 <= first) & (first < 32))
    assert not np.array_equal(first, next_generation)


def test_transition_mask_matches_flattened_population_layout():
    selected = jnp.asarray([1, 3], dtype=jnp.int32)
    mask = np.asarray(transition_admission_mask(
        selected,
        time_size=3,
        population_size=4,
        episode_batch=2,
    )).reshape(3, 4, 2)
    assert mask.shape == (3, 4, 2)
    np.testing.assert_array_equal(mask[:, 0], False)
    np.testing.assert_array_equal(mask[:, 1], True)
    np.testing.assert_array_equal(mask[:, 2], False)
    np.testing.assert_array_equal(mask[:, 3], True)
    assert mask.sum() == 12
