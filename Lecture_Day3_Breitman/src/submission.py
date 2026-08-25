from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


MAX_COMPONENTS = 5


def _get_device(model: nn.Module, device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def _point_predictions(model: nn.Module, loader, device: torch.device):
    model.eval()
    predictions, targets, source_ids = [], [], []

    for features, batch_targets, batch_ids in loader:
        output = model(features.to(device))
        predictions.append(output.detach().cpu().numpy())
        targets.append(batch_targets.numpy())
        source_ids.append(batch_ids.numpy())

    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(source_ids),
    )


@torch.no_grad()
def _mdn_predictions(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_weights, all_means, all_sigmas, all_ids = [], [], [], []

    for features, _targets, batch_ids in loader:
        weights, means, sigmas = model(features.to(device))
        all_weights.append(weights.detach().cpu().numpy())
        all_means.append(means.detach().cpu().numpy())
        all_sigmas.append(sigmas.detach().cpu().numpy())
        all_ids.append(batch_ids.numpy())

    return (
        np.concatenate(all_weights),
        np.concatenate(all_means),
        np.concatenate(all_sigmas),
        np.concatenate(all_ids),
    )


def _empty_submission(source_ids: np.ndarray, max_components: int) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {"source_id": source_ids.astype(np.int64)}
    for component in range(1, max_components + 1):
        data[f"a_{component}"] = np.zeros(len(source_ids), dtype=float)
        data[f"mu_{component}"] = np.zeros(len(source_ids), dtype=float)
        data[f"sigma_{component}"] = np.ones(len(source_ids), dtype=float)
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError(
            "Duplicate source IDs received from the test loader."
        )
    return pd.DataFrame(data)


def _write(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"wrote {output_path} with shape {df.shape}")
    return df


def submission_from_point_model(
    model: nn.Module,
    test_loader,
    valid_loader,
    *,
    output_path: str | Path = "submission_mlp.csv",
    min_sigma: float = 1.0e-3,
    device=None,
) -> pd.DataFrame:
    """Wrap point predictions in a one-component Gaussian mixture.

    The Gaussian width is estimated from validation residuals. Corrupted
    validation rows will therefore worsen the baseline unless students clean them.
    """
    device = _get_device(model, device)
    model.to(device)

    valid_prediction, valid_target, _ = _point_predictions(
        model, valid_loader, device
    )
    residual = valid_target - valid_prediction
    residual = residual[np.isfinite(residual)]
    if not len(residual):
        raise ValueError("No finite validation residuals available")

    sigma = max(float(np.std(residual, ddof=1)), min_sigma)
    print(f"validation residual sigma = {sigma:.6f}")

    test_prediction, _dummy_target, source_ids = _point_predictions(
        model, test_loader, device
    )
    result = _empty_submission(source_ids, MAX_COMPONENTS)
    result["a_1"] = 1.0
    result["mu_1"] = test_prediction
    result["sigma_1"] = sigma
    return _write(result, output_path)


def submission_from_prob_model(
    model: nn.Module,
    test_loader,
    *,
    output_path: str | Path = "submission_mdn.csv",
    device=None,
) -> pd.DataFrame:
    """Write an MDN's predicted mixture parameters in Kaggle format."""
    device = _get_device(model, device)
    model.to(device)

    weights, means, sigmas, source_ids = _mdn_predictions(
        model, test_loader, device
    )
    n_components = weights.shape[1]
    if n_components > MAX_COMPONENTS:
        raise ValueError(
            f"Model has {n_components} components, but Kaggle allows "
            f"at most {MAX_COMPONENTS}"
        )

    result = _empty_submission(source_ids, MAX_COMPONENTS)
    for component in range(n_components):
        index = component + 1
        result[f"a_{index}"] = weights[:, component]
        result[f"mu_{index}"] = means[:, component]
        result[f"sigma_{index}"] = sigmas[:, component]

    return _write(result, output_path)