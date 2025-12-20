# MoE (Mixture of Experts)

This package is organized as **separate MoE variants**, one per load-balancing approach.

Each variant is intentionally self-contained (its own `router.py`, `moe.py`, `load_balancer.py`)
so we don't have to pick config "modes" inside a single implementation.

## Variants

- `variance/`: variance-based aux loss baseline ([README](variance/README.md))
  - Common use: simple baseline / experimentation (not recommended for production)
- `switch/`: Switch-style aux loss (Switch Transformer / GShard / Mixtral family) ([README](switch/README.md))
  - Common use: classic "frequency × importance" auxiliary loss
- `aux_free/`: aux-loss-free bias-based balancing (DeepSeek V2/V3 style) ([README](aux_free/README.md))
  - Common use: update per-expert bias outside autograd
- `global_batch/`: global-batch load balancing (Qwen3 style) ([README](global_batch/README.md))
  - Common use: distributed training with all-reduced expert frequencies

## Quickstart

```python
import torch
from modules.moe import switch

moe = switch.MoE(hidden_dim=16, ffn_dim=32, n_experts=4, topk=2)
y = moe(torch.randn(2, 3, 16))
```
