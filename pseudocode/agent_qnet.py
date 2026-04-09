"""
agent_qnet.py — Per-Agent Utility Network
==========================================
Pseudocode level: 4/10 (conceptual structure, no specific dimensions or activations)

Each agent in the QMIX system has its own independent Q-network (DQN).
The Q-network maps the agent's local observation to utility values for all available actions.
A separate "target" copy of the network provides stable Bellman targets during training.

Key properties:
- Agents are architecturally identical — same network structure, separate parameters
- Each agent sees only its own local observation (no joint state during execution)
- Target network is a frozen copy that updates slowly via soft Polyak averaging
"""


class AgentQNetwork:
    """
    Single-agent Deep Q-Network.

    Maps: local_observation → action_utility_values

    Architecture:
        - Input: agent's local observation (flattened vector)
        - Hidden layers: multiple fully-connected layers with nonlinear activations
        - Output: one utility value per available action

    The output is NOT passed through any activation — Q-values can be
    any real number, positive or negative.
    """

    def __init__(self, observation_dimension, num_actions, hidden_size):
        """
        observation_dimension: size of the flattened local observation vector
        num_actions: number of discrete actions available to this agent
        hidden_size: width of all hidden layers
        """
        # First hidden layer: observation → intermediate representation
        self.layer_1 = FullyConnectedLayer(observation_dimension, hidden_size)

        # Second hidden layer: refines representation
        self.layer_2 = FullyConnectedLayer(hidden_size, hidden_size)

        # Output layer: maps representation to per-action utility values
        # No activation — raw Q-values
        self.output_layer = FullyConnectedLayer(hidden_size, num_actions)

    def forward(self, observation):
        """
        Forward pass: compute Q-values for all actions given local observation.

        observation: tensor of shape (batch_size, observation_dimension)
        returns: tensor of shape (batch_size, num_actions)
        """
        x = apply_activation(self.layer_1(observation))
        x = apply_activation(self.layer_2(x))
        q_values = self.output_layer(x)  # No activation on output
        return q_values

    def get_q_for_action(self, observation, action_indices):
        """
        Convenience method: extract Q-value for specific actions taken.

        observation: (batch_size, observation_dimension)
        action_indices: (batch_size, 1) — which action was taken
        returns: (batch_size, 1) — Q-value for each transition's action
        """
        q_all = self.forward(observation)
        q_selected = gather_along_action_dim(q_all, action_indices)
        return q_selected

    def select_action(self, observation, epsilon):
        """
        Epsilon-greedy action selection.
        Used during training (epsilon > 0) and evaluation (epsilon = 0).

        observation: (1, observation_dimension) — single agent state
        epsilon: float in [0, 1] — probability of random action
        returns: integer action index
        """
        if random_uniform() < epsilon:
            # Exploration: random action
            return random_integer(0, num_actions - 1)
        else:
            # Exploitation: greedy action from Q-network
            q_values = self.forward(observation)
            return argmax(q_values)


class TargetNetwork:
    """
    Frozen copy of AgentQNetwork used for computing stable Bellman targets.

    Updated via Polyak soft averaging after each training epoch:
        target_params ← tau * online_params + (1 - tau) * target_params

    tau << 1 (e.g., ~0.03) ensures the target changes slowly,
    preventing the "moving target" instability in off-policy Q-learning.
    """

    def __init__(self, network_to_copy):
        """
        Create an identical copy of the given network.
        Initially, target_params == online_params.
        """
        self.params = copy_parameters(network_to_copy)

    def soft_update(self, online_network, tau):
        """
        Polyak averaging update.
        tau close to 0: target barely moves (very stable, slow to track learning)
        tau close to 1: target tracks online network closely (less stable)

        Typical range: 0.005–0.05
        """
        for target_param, online_param in zip(self.params, online_network.params):
            target_param = tau * online_param + (1 - tau) * target_param

    def forward(self, observation):
        """Same as AgentQNetwork.forward — used for Bellman target computation."""
        # Uses frozen parameters — no gradient computed through this
        with no_gradient():
            return online_forward_with_frozen_params(observation, self.params)
