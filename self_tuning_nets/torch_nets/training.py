from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_weights(
    model: torch.nn.Module,
    seed: int
) -> Dict[str, torch.Tensor]:
    parameters_size = 0
    for name, tensor in model.state_dict().items():
        parameters_size += np.prod(tensor.size())

    # set weights from uniform distribution in [-1, 1)
    rng = np.random.RandomState(seed)
    seed_weights = 2.0 * rng.random(parameters_size).astype(np.float32) - 1.0

    seed_state_dict = {}
    start, end = 0, 0
    for name, tensor in model.state_dict().items():
        end = start + np.prod(tensor.size())
        new_tensor = torch.from_numpy(
            seed_weights[start:end]).reshape(tensor.size())
        seed_state_dict[name] = new_tensor
        start = end

    model.load_state_dict(seed_state_dict, strict=True)
    return seed_state_dict


def train_step(
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    X: np.ndarray
) -> Tuple[torch.nn.Module, torch.Tensor]:
    model.train()
    pred = model(X)
    loss = nn.MSELoss()(pred, torch.pow(X, 2))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, loss


def eval_step(
    model: torch.nn.Module,
    X: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    pred = model(X)
    loss = nn.MSELoss()(pred, torch.pow(X, 2))
    return loss, pred
