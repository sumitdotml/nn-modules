import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple, Optional

from ...swiglu import SwiGLU as Expert
from .router import MoERouter
from .load_balancer import BiasManager


class RoutingInfo(NamedTuple):
    weights: Tensor  # [total_tokens, topk]
    indices: Tensor  # [total_tokens, topk]


class MoE(nn.Module):
    """
    MoE variant for auxiliary-loss-free load balancing (DeepSeek V2/V3 style).

    This variant caches `(weights, indices)` for computing expert loads after the forward
    pass. A `BiasManager` can be provided to inject a per-expert bias into routing.
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        n_experts: int,
        topk: int,
        bias_manager: Optional[BiasManager] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.topk = topk
        self.bias_manager = bias_manager

        self.router = MoERouter(
            hidden_dim=hidden_dim, n_experts=n_experts, topk=topk, **factory_kwargs
        )
        self.experts = nn.ModuleList(
            [
                Expert(hidden_dim=hidden_dim, ffn_dim=ffn_dim, **factory_kwargs)
                for _ in range(n_experts)
            ]
        )

        self._last_routing_weights: Optional[Tensor] = None
        self._last_routing_indices: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        batch, seqlen, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        bias = self.bias_manager.get_bias() if self.bias_manager is not None else None
        weights, indices = self.router(x, bias=bias)
        self._last_routing_weights = weights
        self._last_routing_indices = indices

        results = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            token_idx, topk_idx = torch.where(indices == i)
            if token_idx.numel() == 0:
                continue
            results[token_idx] += weights[token_idx, topk_idx].unsqueeze(-1) * expert(
                x_flat[token_idx]
            )

        return results.reshape(batch, seqlen, hidden_dim)

    @property
    def last_routing_weights(self) -> Optional[Tensor]:
        return self._last_routing_weights

    @property
    def last_routing_indices(self) -> Optional[Tensor]:
        return self._last_routing_indices

    @property
    def last_routing_info(self) -> Optional[RoutingInfo]:
        weights = self._last_routing_weights
        indices = self._last_routing_indices
        if weights is None or indices is None:
            return None
        return RoutingInfo(weights=weights, indices=indices)

