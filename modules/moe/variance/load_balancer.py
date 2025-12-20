"""
Variance-based load balancing for MoE.

This is a simple baseline: compute the variance of per-expert load fractions and add it
as an auxiliary loss.

Used as: a lightweight baseline (not the common choice in large production MoEs).
"""

import torch
from torch import Tensor


def compute_loss(weights: Tensor, indices: Tensor, n_experts: int) -> Tensor:
    """
    Args:
        weights: Router weights [total_tokens, topk]
        indices: Expert indices [total_tokens, topk]
        n_experts: Total number of experts

    Returns:
        Scalar tensor: variance of expert load distribution.
    """
    total_tokens = weights.shape[0]
    expert_loads = []

    for i in range(n_experts):
        token_idx, topk_idx = torch.where(indices == i)
        expert_weight_sum = weights[token_idx, topk_idx].sum()
        expert_loads.append(expert_weight_sum)

    expert_loads = torch.stack(expert_loads)
    expert_load_fractions = expert_loads / total_tokens
    return torch.var(expert_load_fractions)

