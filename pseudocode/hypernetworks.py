"""
hypernetworks.py — Hypernetwork Modules for QMIX
=================================================
Pseudocode level: 4/10 (conceptual structure, no specific dimensions or activations)

Hypernetworks generate the weights and biases of the QMIX mixing network
dynamically, conditioned on the centralized joint state.

This is the mechanism by which the mixing network becomes state-aware:
rather than fixed weights, the mixing weights shift based on the full
joint observation of both agents — enabling richer, context-dependent
factorization of team value.

The critical constraint:
  Weight matrices must have ALL POSITIVE entries.
  This ensures the mixing network is monotonically increasing in each
  agent's Q-value — the mathematical basis for the IGM property.
  An activation function that guarantees positive outputs (e.g., Softplus)
  is applied to all weight-generating hypernetwork outputs.
  Biases are left unconstrained — they don't affect monotonicity.
"""


class HypernetworkFirstLayerWeights:
    """
    Hypernetwork that generates W₁ — the weight matrix for mixing layer 1.

    Mapping:
        joint_state → weight_matrix_for_layer_1

    Output shape: (batch_size, hidden_dim, num_agents)
        → one weight matrix per batch item
        → applied to the [Q₀, Q₁] vector in mixing layer 1

    Positivity enforcement:
        Output is passed through a strictly-positive activation (e.g., Softplus)
        before being used as a weight matrix.
        This is what ensures ∂Q_tot/∂Qᵢ > 0 for this layer.
    """

    def __init__(self, joint_state_dim, hidden_dim, num_agents):
        """
        joint_state_dim: size of concatenated joint state [s₀; s₁]
        hidden_dim: number of hidden units in mixing layer 1
        num_agents: number of agents (2 for the Overcooked setting)
        """
        # Single linear layer: joint_state → flattened weight values
        # Output size = hidden_dim * num_agents (will be reshaped into a matrix)
        self.linear = FullyConnectedLayer(joint_state_dim, hidden_dim * num_agents)
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

    def forward(self, joint_state):
        """
        joint_state: (batch_size, joint_state_dim)
        returns: (batch_size, hidden_dim, num_agents)
        """
        raw = self.linear(joint_state)

        # Enforce strict positivity — monotonicity requirement
        # Softplus preferred over |·|: differentiable everywhere, no dead-weight region
        positive_weights = strictly_positive_activation(raw)

        # Reshape flat vector into a matrix
        W1 = reshape(positive_weights, shape=(batch_size, self.hidden_dim, self.num_agents))
        return W1


class HypernetworkFirstLayerBias:
    """
    Hypernetwork that generates B₁ — the bias vector for mixing layer 1.

    Biases are unconstrained (no positivity requirement).
    They shift the mixing output without affecting the gradient with
    respect to agent Q-values, so they don't disturb monotonicity.

    Output shape: (batch_size, 1, hidden_dim)
    """

    def __init__(self, joint_state_dim, hidden_dim):
        self.linear = FullyConnectedLayer(joint_state_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, joint_state):
        """returns: (batch_size, 1, hidden_dim)"""
        B1 = self.linear(joint_state)
        return reshape(B1, shape=(batch_size, 1, self.hidden_dim))


class HypernetworkSecondLayerWeights:
    """
    Hypernetwork that generates W₂ — the weight matrix for mixing layer 2.

    This layer collapses the hidden representation to a scalar (Q_tot).
    Output shape: (batch_size, 1, hidden_dim)
        → one row vector per batch item
        → applied to the hidden representation h in mixing layer 2

    Positivity enforcement:
        Same as W₁ — Softplus applied to guarantee ∂Q_tot/∂h > 0,
        which combined with W₁ positivity gives ∂Q_tot/∂Qᵢ > 0 overall.
    """

    def __init__(self, joint_state_dim, hidden_dim):
        self.linear = FullyConnectedLayer(joint_state_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, joint_state):
        """returns: (batch_size, 1, hidden_dim)"""
        raw = self.linear(joint_state)
        positive_weights = strictly_positive_activation(raw)
        W2 = reshape(positive_weights, shape=(batch_size, 1, self.hidden_dim))
        return W2


class HypernetworkSecondLayerBias:
    """
    Hypernetwork that generates B₂ — the scalar output bias.

    Maps joint_state → scalar, one per batch item.
    Unconstrained. Provides a state-dependent global offset to Q_tot,
    improving expressivity without affecting monotonicity.

    Output shape: (batch_size, 1, 1)
    """

    def __init__(self, joint_state_dim):
        # Output dim = 1: a single scalar per batch item
        self.linear = FullyConnectedLayer(joint_state_dim, 1)

    def forward(self, joint_state):
        """returns: (batch_size, 1, 1)"""
        B2 = self.linear(joint_state)
        return reshape(B2, shape=(batch_size, 1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Conceptual note on the Softplus vs. Absolute Value choice
# ─────────────────────────────────────────────────────────────────────────────
#
# Original QMIX paper: uses |·| (hard absolute value) for weight positivity
# This implementation: uses Softplus = log(1 + exp(x))
#
# Why Softplus is preferable:
#   1. Differentiable everywhere — |·| has zero gradient at 0
#   2. No "dead weight" neurons — weights passing through 0 can recover gradient
#   3. Smoother loss landscape during the high-variance early training phase
#   4. Strictly positive for all real inputs (Softplus output > 0 always)
#
# Both enforce monotonicity. Softplus does so with better gradient properties.
