import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class MoERouter(nn.Module):
    r"""
    Mixture of Experts Router. Routes each token to the top-k most suitable experts
    based on learned routing weights, enabling sparse activation in MoE architectures.

    On a high level, the router creates a learned specialization where different experts become
    proficient at different types of tokens (e.g., physics vs. geography content),
    enabling efficient sparse computation while maintaining model capacity.

    The router performs the following operations:
    ```
    scores = W_router · x                    (compute affinity for each expert)
    probabilities = softmax(scores)          (normalize across all experts)
    top_k_probs, top_k_indices = topk(probabilities, k)  (select best k experts)
    weights = renormalize(top_k_probs)       (ensure selected weights sum to 1)
    ```

    Process:
    1. **Scoring**: Projects input tokens to expert scores via learned linear layer
    2. **Softmax**: Converts scores to probabilities (ensures all 8 sum to 1.0)
    3. **Top-k Selection**: Picks k experts with highest probabilities per token
    4. **Renormalization**: Rescales selected weights to sum to 1.0

    Args:
        hidden_dim: Dimension of input token representations (e.g., 4096 in Mixtral)
        n_ffn_experts: Total number of available experts (e.g., 8 in Mixtral)
        topk: Number of experts to activate per token (e.g., 2 in Mixtral)
        device: Device to place router parameters on
        dtype: Data type for router parameters

    Shapes:
        Input: [batch_size, seq_len, hidden_dim]
        Output weights: [batch_size * seq_len, topk]
        Output indices: [batch_size * seq_len, topk]

    Returns:
        tuple: (weights, indices) where:
            - weights: Normalized routing weights for selected experts [total_tokens, topk]
            - indices: Expert indices selected for each token [total_tokens, topk]

    Example:
        >>> router = MoERouter(hidden_dim=4096, n_ffn_experts=8, topk=2)
        >>> x = torch.randn(2, 10, 4096)  # 2 sequences, 10 tokens each
        >>> weights, indices = router(x)
        >>> weights.shape  # [20, 2] - each of 20 tokens routes to 2 experts
        >>> indices.shape  # [20, 2]
        >>> weights.sum(dim=-1)  # All close to 1.0 (renormalized)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_ffn_experts: int,
        topk: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.topk = topk
        self.W_router = nn.Linear(
            in_features=hidden_dim,
            out_features=n_ffn_experts,
            bias=False,
            **factory_kwargs,
        )

    # x is [batch, seqlen, hidden_dim]
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        being a bit comprehensive & verbose here as I'm learning it all, might clean this
        up later by for instance, getting rid of unused names & comments. but for now it's chill
        """
        # Step 1: flattening batch and sequence dimensions
        batch, seqlen, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # [batch*seqlen, hidden_dim]

        # Step 2: raw scores' computation for all experts
        router_matrix = self.W_router(x_flat)  # [batch*seqlen, n_ffn_experts]

        # Step 3: softmax
        probabilities = torch.softmax(router_matrix, dim=1)

        # Step 4: top-k experts' selection
        # values: [batch*seqlen, topk], indices: [batch*seqlen, topk]
        values, indices = torch.topk(probabilities, k=self.topk)

        # Step 5: weights' renormalization
        values = MoERouter._renormalization(values)

        return values, indices

    @staticmethod
    def _renormalization(input: Tensor) -> Tensor:
        total = input.sum(
            dim=-1, keepdim=True
        )  # total sum of experts' raw scores, not token count, hence -1 dim
        renormalized = input / total  # [batch*seqlen, topk]
        return renormalized


class MoELayer(nn.Module):
    def __init__(
        self,
        n_experts: int,
        topk: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

    def forward(self, x: Tensor) -> Tensor:
        pass
        "contd..."
