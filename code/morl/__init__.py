"""
MORL: Multi-Objective Reinforcement Learning for sequential food recommendation.

Operates on frozen user/item embeddings produced by the SGSL+MGDA training phase.
No modifications are made to the SGSL training pipeline.

Modules:
    environment  – deterministic K-step MDP for sequential list construction.
    policy       – conditional policy network π(a | s, w).
    training     – REINFORCE-based MORL training loop.
    morl_main    – entry point: loads checkpoint, trains policy, evaluates.
"""
