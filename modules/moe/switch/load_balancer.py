"""
Switch Transformer / Mixtral auxiliary load balancing loss.

This is the standard aux loss used in the Switch Transformer / GShard / Mixtral family.

NOTE: The actual loss is intentionally left as a TODO scaffold in this repo.
"""

import torch
from torch import Tensor


def compute_loss(
    router_logits: Tensor, indices: Tensor, n_experts: int, topk: int
) -> Tensor:
    """
    Computes Switch Transformer style auxiliary load balancing loss.

    Args:
        router_logits: Raw router scores before softmax [total_tokens, n_experts]
        indices: Selected expert indices [total_tokens, topk]
        n_experts: Total number of experts
        topk: Number of experts selected per token

    Returns:
        Scalar tensor representing the load balancing loss.

    TODO: implement the actual computation
        1. Compute router probabilities from logits (softmax)
        2. Compute f_i: fraction of tokens where expert i is in top-k
        3. Compute P_i: mean router probability for expert i across all tokens
        4. Return n_experts * sum(f_i * P_i)
    """
    raise NotImplementedError(
        "Switch Transformer load balancing loss not yet implemented. "
        "See docstring for the algorithm."
    )

