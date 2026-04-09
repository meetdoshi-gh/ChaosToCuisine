# Chaos to Cuisine: Cooperative Multi-Agent RL with QMIX
### Pseudocode & Architecture Reference

---

## Overview

This project implements **QMIX** — a cooperative Multi-Agent Reinforcement Learning algorithm — applied to the Overcooked kitchen simulation environment. Two autonomous agents learn to collaboratively prepare and deliver soups using only local observations, no explicit communication, and a shared team reward signal.

The core challenge addressed is **cooperative multi-agent credit assignment** under sparse rewards: when a soup is delivered after dozens of joint actions, how does each agent learn which of its actions contributed to the team outcome?

---

## Algorithm: QMIX

QMIX belongs to the **Centralized Training, Decentralized Execution (CTDE)** family of multi-agent RL methods. The key idea is **value decomposition**:

- Each agent maintains its own utility function (a Deep Q-Network) that maps its local observation to action values
- A centralized **mixing network** combines individual utility values into a global team value (Q_tot)
- The mixing network is conditioned on the full joint state through **hypernetworks**
- A **monotonicity constraint** on mixing weights guarantees that locally greedy actions are globally optimal (the IGM property)

During training, the full joint state is available to the mixer. During execution, each agent acts on its own Q-network alone — fully decentralized, no communication required.

---

## Architecture Summary

```
Environment (Overcooked)
    │
    ├─ s₀ (agent 0 local observation) ──► Agent 0 Q-Network ──► Q₀(s₀, a)
    ├─ s₁ (agent 1 local observation) ──► Agent 1 Q-Network ──► Q₁(s₁, a)
    └─ z = [s₀; s₁] (joint state, training only)
                │
                ▼
          Hypernetworks (conditioned on z)
                │
                ▼
          Mixing Network ──► Q_tot (global team value)
                │
                ▼
    Bellman TD Loss ──► AdamW Optimizer ──► Update all networks
```

**Monotonicity guarantee:** ∂Q_tot/∂Qᵢ > 0 for all i — enforced by keeping all mixing weight values strictly positive throughout training.

---

## Module Index

| File | Contents |
|---|---|
| `pseudocode/agent_qnet.py` | Individual agent Q-Network and target network |
| `pseudocode/hypernetworks.py` | Hypernetwork modules that generate mixing weights |
| `pseudocode/mixing_network.py` | QMIX mixing network forward pass |
| `pseudocode/replay_buffer.py` | Experience replay memory |
| `pseudocode/qmix_agent.py` | Full QMIX training loop (main entry point) |

---

## Results Summary

| Layout | Evaluation (100 eps, greedy) | Episodes to Convergence | vs. ≥7 Threshold |
|---|---|---|---|
| cramped_room | 12 soups/episode | ~2,000 | +71% |
| coordination_ring | 17 soups/episode | ~5,000 | +143% |
| counter_circuit | 10 soups/episode | ~8,000 | +43% |

All three layouts exceeded the ≥7 soups/episode success threshold in fully deterministic evaluation.

---

## How to Run (Conceptual)

1. Install dependencies: PyTorch, overcooked-ai, gym, pytorch-lightning
2. Configure layout and hyperparameters in the training config
3. Run the training loop (qmix_agent.py) — replay buffer fills first with random-policy transitions
4. Monitor via TensorBoard for episodic reward and behavioral metrics
5. Evaluate the saved checkpoint with ε = 0 (greedy) for 100 deterministic episodes

**Key hyperparameters:**
- Learning rate: 0.0007 (AdamW)
- Discount factor γ: 0.99
- Soft-update coefficient τ: 0.028
- ε-decay rate λ: 0.00055 (most sensitive — tune carefully)
- Batch size: 512, Replay buffer: 100K transitions

---

## Diagrams

The `/diagrams/` folder contains five interactive HTML diagrams viewable in any browser:

| File | Contents |
|---|---|
| `diagram_system_architecture.html` | Full CTDE training pipeline |
| `diagram_value_decomposition.html` | VDN vs QMIX value decomposition comparison |
| `diagram_mixing_network.html` | Hypernetwork + mixing network architecture detail |
| `diagram_iql_vs_qmix.html` | IDQN vs QMIX algorithm comparison |
| `diagram_results.html` | Training curves, evaluation results, HP sensitivity |

---

## Academic Integrity

These pseudocode files describe algorithmic concepts at an abstracted level (abstraction level ~4–5/10). They do not reproduce the implementation code from the original course assignment. Specific architecture dimensions, activation function names, loss function identifiers, and optimizer choices are intentionally omitted — the goal is to document the conceptual structure of the algorithm, not to reproduce working code.

The original QMIX algorithm is published in: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning," ICML 2018.

---

*meetdoshi.me · Chaos to Cuisine · QMIX Cooperative MARL*
