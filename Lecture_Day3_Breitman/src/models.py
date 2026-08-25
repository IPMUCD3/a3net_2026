import torch
from torch import nn
from torch.nn import functional as F


def _make_mlp(
    n_features,
    hidden_dims,
    dropout,
    output_dim):
    layers: list[nn.Module] = []
    in_dim = n_features

    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim

    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class MLPRegressor(nn.Module):
    """Point-estimate baseline: photometric features -> one redshift."""

    def __init__(
        self,
        n_features,
        hidden_dims= (32,),
        dropout = 0.0,
    ):
        super().__init__()
        self.network = _make_mlp(n_features, hidden_dims, dropout, output_dim=1)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MDN(nn.Module):
    """Conditional one-dimensional Gaussian mixture model."""

    def __init__(
        self,
        n_features,
        hidden_dims = (32,),
        n_components = 3,
        dropout = 0.0,
        min_sigma = 1.0e-3):
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be at least 1")

        self.n_components = n_components
        self.min_sigma = min_sigma
        self.network = _make_mlp(
            n_features,
            hidden_dims,
            dropout,
            output_dim=3 * n_components,
        )

    def forward(self, x):
        raw = self.network(x)
        logits, means, raw_scales = torch.chunk(raw, 3, dim=-1)

        weights = torch.softmax(logits, dim=-1)
        sigmas = F.softplus(raw_scales) + self.min_sigma
        return weights, means, sigmas


def mdn_nll(output, target):
    """Mean negative log likelihood of a one-dimensional Gaussian mixture."""
    weights, means, sigmas = output
    target = target.unsqueeze(-1)

    log_component_density = torch.distributions.Normal(
        means, sigmas
    ).log_prob(target)
    log_mixture = torch.log(weights.clamp_min(1.0e-12)) + log_component_density
    return -torch.logsumexp(log_mixture, dim=-1).mean()
