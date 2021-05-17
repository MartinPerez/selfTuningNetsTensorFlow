import numpy as np

from self_tuning_nets.visualization import (trajectories_dist_from_target,
                                            trajectories_dist_matrix)


def test_dist_matrix():
    result = trajectories_dist_matrix([
            [np.array([3.0, 2.0, 3.0]),
             np.array([1.0, 0.0, 1.0])],
            [np.array([4.0, 3.0, 4.0]),
             np.array([3.0, 2.0, 3.0]),
             np.array([1.0, 0.0, 1.0])]
        ],
        np.array([1.0, 0.0, 1.0]),
        100)

    expected_dist = np.array(
        [
            [0., 6., 3., 0., 6., 6.],
            [6., 0., 9., 6., 0., 0.],
            [3., 9., 0., 3., 9., 9.],
            [0., 6., 3., 0., 6., 6.],
            [6., 0., 9., 6., 0., 0.],
            [6., 0., 9., 6., 0., 0.]
        ])
    np.testing.assert_equal(result[0], expected_dist)
    assert result[1] == [2, 3, 1]


def test_trajectories_dist_from_target():
    result = trajectories_dist_from_target([
            [np.array([3.0, 2.0, 3.0]),
             np.array([1.0, 0.0, 1.0])],
            [np.array([4.0, 3.0, 4.0]),
             np.array([3.0, 2.0, 3.0]),
             np.array([1.0, 0.0, 1.0])]
        ],
        np.array([1.0, 0.0, 1.0]),
        100)

    assert len(result) == 2
    np.testing.assert_equal(result[0], np.array([6., 0.]))
    np.testing.assert_equal(result[1], np.array([9., 6., 0.]))
