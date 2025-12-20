"""
Global-batch load balancing (Qwen3 style).

This approach computes expert selection frequencies over the *global batch* (via
all-reduce) instead of forcing balance within each micro-batch.

NOTE: Distributed sync + the loss itself are intentionally left as TODO scaffolds.
"""

import torch
from torch import Tensor
from typing import Optional


def compute_local_frequencies(indices: Tensor, n_experts: int) -> Tensor:
    """
    Computes expert selection frequency within a local micro-batch.

    Args:
        indices: Expert indices [total_tokens, topk]
        n_experts: Total number of experts

    Returns:
        Tensor of shape [n_experts] with selection counts for each expert.

    TODO: implement frequency counting
    """
    raise NotImplementedError("Local frequency computation not yet implemented.")


def sync_frequencies(local_freqs: Tensor, process_group: Optional[object] = None) -> Tensor:
    """
    Synchronizes expert frequencies across all distributed processes.

    Args:
        local_freqs: Local expert frequencies [n_experts]
        process_group: Distributed process group (None for default)

    Returns:
        Global expert frequencies after all-reduce.

    TODO: implement distributed all-reduce
    """
    raise NotImplementedError(
        "Distributed frequency sync not yet implemented. "
        "Use torch.distributed.all_reduce when implementing."
    )


def compute_loss(global_freqs: Tensor, router_probs: Tensor, n_experts: int) -> Tensor:
    """
    Computes global-batch load balancing loss.

    Args:
        global_freqs: Global expert selection frequencies [n_experts]
        router_probs: Router probabilities [total_tokens, n_experts]
        n_experts: Total number of experts

    Returns:
        Scalar load balancing loss.

    TODO: implement the loss computation following Switch Transformer formulation
          but with global frequencies instead of micro-batch frequencies.
    """
    raise NotImplementedError(
        "Global-batch load balancing loss not yet implemented. "
        "See docstring for the algorithm."
    )

