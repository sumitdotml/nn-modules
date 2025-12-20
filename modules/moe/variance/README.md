# MoE: Variance-Based Load Balancing

This is a small MoE implementation wired for **variance-based** load balancing.

## When to use

- To be very honest, we should not use this load balancing method in production. I wrote this mainly to understand the concept of load balancing and how to implement it during my first time learning about MoE.

## Common names / references

- “variance-based load balancing” (simple baseline; not a standard paper loss)

## API

- Router returns `(weights, indices)` (top-k only).
- MoE caches `last_routing_weights` / `last_routing_indices` for `load_balancer.compute_loss(...)`.

