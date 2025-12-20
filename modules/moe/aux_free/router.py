import torch
import torch.nn as nn
from torch import Tensor
from typing import NamedTuple, Optional


class RouterOutput(NamedTuple):
    weights: Tensor  # [total_tokens, topk] - renormalized weights for selected experts
    indices: Tensor  # [total_tokens, topk] - selected expert indices per token


class MoERouter(nn.Module):
    """
    Top-k router with an optional per-expert bias term (aux-loss-free balancing).

    DeepSeek-style auxiliary-loss-free balancing adjusts a bias vector over time and
    adds it to the router logits before top-k selection.
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

    def forward(self, x: Tensor, *, bias: Optional[Tensor] = None) -> RouterOutput:
        _, _, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # [total_tokens, hidden_dim]

        logits = self.W_router(x_flat)  # [total_tokens, n_experts]
        if bias is not None:
            logits = logits + bias.to(dtype=logits.dtype, device=logits.device)

        probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=self.topk, dim=-1)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        return RouterOutput(topk_weights, topk_indices)

