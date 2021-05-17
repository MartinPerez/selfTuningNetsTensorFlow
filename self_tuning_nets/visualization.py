from typing import Callable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera
from matplotlib import rc
from matplotlib.animation import ArtistAnimation
from tqdm.notebook import tqdm

rc('animation', html='html5')


def function_animation(
    X_eval: np.ndarray,
    function_trajectories: List[List[np.ndarray]],
    line_palette: List  # matplotlib color arguments
) -> ArtistAnimation:
    x = X_eval
    fig = plt.figure()
    camera = Camera(fig)
    for loop_idx in tqdm(range(np.max(
            [len(t) for t in function_trajectories]))):
        plt.plot(x, x**2, color="black", linewidth=2)
        for trajectory, color in zip(function_trajectories, line_palette):
            pred_idx = min(loop_idx, len(trajectory) - 1)
            pred = trajectory[pred_idx]
            y = pred
            plt.plot(x, y, color=color, linestyle="dashed")
        camera.snap()
    plt.close()
    return camera.animate()


FUNC_DIST = Callable[[np.ndarray, np.ndarray], np.float]


def abs_diff_sum(x1: np.ndarray, x2: np.ndarray) -> np.float:
    return np.sum(np.abs(x1 - x2))


def trajectories_dist_matrix(
    function_trajectories: List[List[np.ndarray]],
    target_predictions: np.ndarray,
    dist_cap: float,
    func_dist: FUNC_DIST = abs_diff_sum,
    verbose=0
) -> Tuple[np.ndarray, List[int]]:
    flattened = [p for ft in function_trajectories for p in ft]
    for pred in flattened:
        assert pred.shape == target_predictions.shape
    flattened += [target_predictions]
    # add target function to matrix
    nb_per_trajectory = [len(t) for t in function_trajectories] + [1]
    nb_pred = np.sum(nb_per_trajectory)  # every 100 batches

    dist_matrix = np.zeros((nb_pred, nb_pred))
    for row, row_pred in tqdm(
            enumerate(flattened), total=len(flattened), unit='row'):
        for col, col_pred in enumerate(flattened):
            # We need to cap distances for a visually normalized matrix
            dist_matrix[row][col] = min(
                func_dist(row_pred, col_pred), dist_cap)

    if verbose > 0:
        print(f"In total have {nb_pred} predictions including "
              f"last the actual function")
        print(f"trajectories have {nb_per_trajectory} points")
        print(f"dist_matrix have shape {dist_matrix.shape}")
    return dist_matrix, nb_per_trajectory


def trajectories_plot_from_matrix(
    t_dist: np.ndarray,
    nb_per_trajectory: List[int],
    lines_palette: List,  # matplotlib color arguments
    SAMPLING_SPEED: int,
    title_extra: str = ""
) -> None:
    accum_idx = np.cumsum(nb_per_trajectory).tolist()
    for start, end, color in tqdm(
            zip([0] + accum_idx[:-1], accum_idx, lines_palette)):
        plt.plot(range(end - start), t_dist[-1, start:end], color=color)

    plt.xlabel(f"sample every {SAMPLING_SPEED} batches")
    plt.ylabel("Function distance from x^2")
    plt.title(f"Function trajectories {title_extra}\n"
              f"(stable distance imply space sinks)")


def trajectories_dist_from_target(
    function_trajectories: List[List[np.ndarray]],
    target_predictions: np.ndarray,
    dist_cap: float,
    func_dist: FUNC_DIST = abs_diff_sum,
    verbose=0
) -> List[np.ndarray]:
    trajectories_distances = []
    for trajectory in function_trajectories:
        function_distances = []
        for pred in trajectory:
            dist = min(func_dist(target_predictions, pred), dist_cap)
            function_distances.append(dist)
        trajectories_distances.append(function_distances)
    return trajectories_distances


def trajectories_plot(
    trajectories_distances: List[np.ndarray],
    lines_palette: List,  # matplotlib color arguments
    SAMPLING_SPEED: int,
    title_extra: str = ""
) -> None:
    for distances, color in tqdm(zip(trajectories_distances, lines_palette)):
        plt.plot(distances, color=color)

    plt.xlabel(f"sample every {SAMPLING_SPEED} batches")
    plt.ylabel("Function distance from x^2")
    plt.title(f"Function trajectories {title_extra}\n"
              f"(stable distance imply staying in local minima)")


def trajectories_general_plot(
    trajectories: List[np.ndarray],
    lines_palette: List,  # matplotlib color arguments
    title: str = "",
    xlabel: str = "Epoch",
    ylabel: str = ""
) -> None:
    for distances, color in zip(trajectories, lines_palette):
        plt.plot(distances, color=color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def trajectories_legend(labels, lines_palette):
    palette = dict(zip(labels, lines_palette))
    handles = [mpl.patches.Patch(
        color=palette[x], label=x) for x in palette.keys()]
    plt.legend(handles=handles)
    plt.gca().set_axis_off()
