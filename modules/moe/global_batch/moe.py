import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple, Optional

from ...swiglu import SwiGLU as Expert
from .router import MoERouter


class RoutingInfo(NamedTuple):
    indices: Tensor  # [total_tokens, topk]
    probs: Tensor  # [total_tokens, n_experts]


class MoE(nn.Module):
    """
    MoE variant for global-batch load balancing (Qwen3-style).

    This variant caches:
    - local routing `indices` (for local frequency counting)
    - router probabilities via cached logits (for "importance" statistics)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        n_experts: int,
        topk: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.hidden_dim = hidden_dim
        self.n_experts = n_experts
        self.topk = topk

        self.router = MoERouter(
            hidden_dim=hidden_dim, n_experts=n_experts, topk=topk, **factory_kwargs
        )
        self.experts = nn.ModuleList(
            [
                Expert(hidden_dim=hidden_dim, ffn_dim=ffn_dim, **factory_kwargs)
                for _ in range(n_experts)
            ]
        )

        self._last_routing_indices: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        batch, seqlen, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)

        weights, indices = self.router(x)
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
    def last_routing_indices(self) -> Optional[Tensor]:
        return self._last_routing_indices

    @property
    def last_router_probs(self) -> Optional[Tensor]:
        return self.router.last_probs

    @property
    def last_routing_info(self) -> Optional[RoutingInfo]:
        indices = self._last_routing_indices
        probs = self.router.last_probs
        if indices is None or probs is None:
            return None
        return RoutingInfo(indices=indices, probs=probs)

