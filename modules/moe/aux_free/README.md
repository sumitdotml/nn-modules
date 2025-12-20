# MoE: Aux-Loss-Free (DeepSeek V2/V3 Style)

This is an MoE implementation wired for **auxiliary-loss-free** load balancing.

## When to use

- When we want to avoid auxiliary-loss gradients and instead balance experts by updating a
  **bias vector** outside autograd and adding it to the router logits.

## Common names / references

- DeepSeek-V2 / DeepSeek-V3 “aux-loss-free load balancing”

## API

- `load_balancer.BiasManager` stores the per-expert bias (update logic is TODO in this repo).
- `MoE(..., bias_manager=...)` applies `bias_manager.get_bias()` during routing.
- MoE caches `last_routing_weights` / `last_routing_indices` so we can compute expert loads
  after the forward pass.

