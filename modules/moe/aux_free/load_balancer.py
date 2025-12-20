"""
Auxiliary-loss-free load balancing (DeepSeek V2/V3 style).

DeepSeek uses a bias-based balancing strategy instead of an auxiliary loss:
we update a per-expert bias term outside autograd and add it to router logits.

NOTE: The update logic and load computation are intentionally left as TODO scaffolds.
"""

import torch
from torch import Tensor
from typing import Optional


class BiasManager:
    def __init__(
        self,
        n_experts: int,
        gamma: float = 0.001,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.n_experts = n_experts
        self.gamma = gamma
        self.bias = torch.zeros(n_experts, device=device, dtype=dtype)
        self.target_load = 1.0 / n_experts

    def get_bias(self) -> Tensor:
        return self.bias

    def update(self, expert_loads: Tensor) -> None:
        """
        Args:
            expert_loads: Fraction of tokens routed to each expert [n_experts]

        TODO: implement the bias update logic
            - Increase bias for underloaded experts
            - Decrease bias for overloaded experts
        """
        raise NotImplementedError(
            "Bias update logic not yet implemented. " "See docstring for the algorithm."
        )


def compute_expert_loads(weights: Tensor, indices: Tensor, n_experts: int) -> Tensor:
    """
    Computes the load fraction for each expert.

    Args:
        weights: Router weights [total_tokens, topk]
        indices: Expert indices [total_tokens, topk]
        n_experts: Total number of experts

    Returns:
        Tensor of shape [n_experts] with load fraction for each expert.

    TODO: implement load computation
    """
    raise NotImplementedError("Expert load computation not yet implemented.")

