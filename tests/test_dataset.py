from numpy import testing

from self_tuning_nets.dataset import (standard_normal_batches,
                                      x_interval_batches)


def test_deterministic_batch_continuation():
    batches = list(x_interval_batches(1, max_batches=3, batch_size=3))
    print(list(batches))
    assert len(batches) == 3
    assert len(batches[0]) == 3

    batches2 = list(x_interval_batches(1, max_batches=4, batch_size=3))
    # same first 3 batches if we only change number of batches
    for b1, b2 in zip(batches, batches2):
        testing.assert_almost_equal(b1, b2)


def test_batch_seeds_difference():
    batches = list(x_interval_batches(1, max_batches=3, batch_size=3))
    batches2 = list(x_interval_batches(2, max_batches=3, batch_size=3))
    for b1, b2 in zip(batches, batches2):
        testing.assert_raises(
            AssertionError, testing.assert_array_equal, b1, b2)


def test_deterministic_batch_continuation_normal():
    batches = list(standard_normal_batches(1, max_batches=3, batch_size=3))
    print(list(batches))
    assert len(batches) == 3
    assert len(batches[0]) == 3

    batches2 = list(standard_normal_batches(1, max_batches=4, batch_size=3))
    # same first 3 batches if we only change number of batches
    for b1, b2 in zip(batches, batches2):
        testing.assert_almost_equal(b1, b2)


def test_batch_seeds_difference_normal():
    batches = list(standard_normal_batches(1, max_batches=3, batch_size=3))
    batches2 = list(standard_normal_batches(2, max_batches=3, batch_size=3))
    for b1, b2 in zip(batches, batches2):
        testing.assert_raises(
            AssertionError, testing.assert_array_equal, b1, b2)
