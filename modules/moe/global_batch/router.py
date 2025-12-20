import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple, Optional


class RouterOutput(NamedTuple):
    weights: Tensor  # [total_tokens, topk] - renormalized weights for selected experts
    indices: Tensor  # [total_tokens, topk] - selected expert indices per token


class MoERouter(nn.Module):
    """
    Top-k router for global-batch load balancing (Qwen3-style).

    This router caches dense logits from the last forward pass so we can compute
    full router probabilities and global-batch "importance" statistics.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_experts: int,
        topk: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.topk = topk
        self.W_router = nn.Linear(hidden_dim, n_experts, bias=False, **factory_kwargs)

        self._last_logits: Optional[Tensor] = None

    def forward(self, x: Tensor) -> RouterOutput:
        _, _, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # [total_tokens, hidden_dim]

        logits = self.W_router(x_flat)  # [total_tokens, n_experts]
        self._last_logits = logits

        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=self.topk, dim=-1)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return RouterOutput(topk_weights, topk_indices)

    @property
    def last_logits(self) -> Optional[Tensor]:
        return self._last_logits

    @property
    def last_probs(self) -> Optional[Tensor]:
        logits = self._last_logits
        if logits is None:
            return None
        return torch.softmax(logits, dim=-1)

