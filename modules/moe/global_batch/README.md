# MoE: Global-Batch Load Balancing (Qwen3 Style)

This is an MoE implementation wired for **global-batch** load balancing.

## When to use

- When we are training with small micro-batches and want to avoid “forcing balance within each
  micro-batch”, which can harm expert specialization.
- When we are in distributed training and can all-reduce per-expert selection frequencies to
  approximate global-batch statistics.

## Common names / references

- Qwen3 “global-batch load balance”
- Same Switch-style objective, but with global (synced) frequencies

## API

- MoE caches `last_routing_indices` and exposes `last_router_probs` (computed from cached logits).
- `load_balancer.*` functions are scaffolds (TODOs) for frequency counting, all-reduce, and loss.

