from __future__ import annotations

import copy
from collections.abc import Callable

import torch
from torch import nn

LossFunction = Callable[[object, torch.Tensor], torch.Tensor]


def _mean_loss(
    model: nn.Module,
    loader,
    loss_fn: LossFunction,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_examples = 0

    for features, targets, _source_ids in loader:
        features = features.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            output = model(features)
            loss = loss_fn(output, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = features.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    loss_fn: LossFunction,
    *,
    learning_rate: float = 1.0e-3,
    n_epochs: int = 20,
    patience: int = 5,
    device: str | torch.device | None = None,
) -> dict[str, list[float]]:
    """Train with validation, early stopping, and best-weight restoration."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {"train_loss": [], "valid_loss": []}
    best_valid = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, n_epochs + 1):
        train_loss = _mean_loss(
            model, train_loader, loss_fn, device, optimizer=optimizer
        )
        with torch.no_grad():
            valid_loss = _mean_loss(
                model, valid_loader, loss_fn, device, optimizer=None
            )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        print(
            f"epoch {epoch:02d}/{n_epochs} | "
            f"train={train_loss:.5f} | valid={valid_loss:.5f}"
        )

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stopping after epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return history