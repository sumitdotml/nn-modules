# MoE: Switch-Style Auxiliary Loss (Switch / GShard / Mixtral)

This is an MoE implementation wired for the **Switch-style auxiliary load balancing loss**.

## When to use

- When we want the classic aux loss that uses:
  - **frequency**: how often each expert is selected (from `indices`)
  - **importance**: mean probability mass per expert (from `softmax(logits)`)
- Typical in the Switch Transformer / GShard / Mixtral family.

## Common names / references

- Switch Transformer (Fedus et al., 2021)
- GShard (Lepikhin et al., 2020)
- Mixtral (Mistral AI, 2024) uses the same aux-loss family with top-2 routing

## API

- Router caches `last_logits` (dense scores for all experts).
- MoE caches `last_routing_indices`.
- `load_balancer.compute_loss(router_logits, indices, n_experts, topk)` is a scaffold (TODO).

