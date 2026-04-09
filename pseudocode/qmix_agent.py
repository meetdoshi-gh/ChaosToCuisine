"""
qmix_agent.py — Main QMIX Training Loop
=========================================
Pseudocode level: 4/10 (conceptual structure, no implementation details)

This is the top-level training orchestrator for the QMIX algorithm.
It coordinates all components:
    - Two agent Q-networks (online + target for each)
    - Mixing network (online + target copy)
    - Hypernetworks (part of mixing network)
    - Replay buffer
    - Environment interaction loop
    - Gradient computation and parameter updates
    - Soft target network updates
    - Epsilon decay schedule

Training is epoch-based (one episode = one epoch in PyTorch Lightning).
Within each epoch, multiple gradient update steps occur over sampled batches.
At the end of each epoch:
    - Target networks are soft-updated
    - Epsilon is decayed
    - Training metrics are logged

Evaluation is run separately, with epsilon = 0 (fully greedy policy).
"""

from agent_qnet import AgentQNetwork, TargetNetwork
from mixing_network import QMIXMixingNetwork
from replay_buffer import ReplayBuffer


class QMIXAgent:
    """
    Top-level QMIX training agent.
    Wraps all networks, the replay buffer, and the training/evaluation logic.
    """

    def __init__(self, config):
        """
        config: dictionary of all hyperparameters

        Key config fields:
            observation_dim: size of each agent's local observation
            num_actions: number of discrete actions
            joint_state_dim: observation_dim * num_agents (centralized info)
            hidden_dim: width of agent network hidden layers
            mixing_hidden_dim: width of mixing network hidden layer
            buffer_capacity: max replay buffer size
            batch_size: number of transitions per gradient update
            learning_rate: optimizer step size
            gamma: discount factor for future rewards
            tau: soft-update coefficient for target networks
            epsilon_min: minimum exploration probability
            epsilon_decay_rate: exponential decay rate for epsilon
            num_agents: number of cooperating agents (2 for Overcooked)
        """
        # ── Per-agent networks (online + target) ──────────────────────────────
        self.q_network_agent0 = AgentQNetwork(
            config.observation_dim, config.num_actions, config.hidden_dim
        )
        self.q_network_agent1 = AgentQNetwork(
            config.observation_dim, config.num_actions, config.hidden_dim
        )
        self.target_q_agent0 = TargetNetwork(self.q_network_agent0)
        self.target_q_agent1 = TargetNetwork(self.q_network_agent1)

        # ── Mixing network (online + target copy) ─────────────────────────────
        self.mixing_network = QMIXMixingNetwork(
            config.joint_state_dim, config.num_agents, config.mixing_hidden_dim
        )
        self.target_mixing_network = TargetNetwork(self.mixing_network)

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(config.buffer_capacity)

        # ── Optimizer: single optimizer covers all online network parameters ──
        all_online_params = collect_parameters([
            self.q_network_agent0,
            self.q_network_agent1,
            self.mixing_network,
        ])
        self.optimizer = Optimizer(all_online_params, lr=config.learning_rate)

        # ── Exploration state ─────────────────────────────────────────────────
        self.epsilon = 1.0  # Start fully exploratory
        self.episode_count = 0

        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # Core training step
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self):
        """
        One episode of environment interaction + one or more gradient updates.
        Called repeatedly by the training loop.
        """
        # ── 1. Interact with environment for one full episode ────────────────
        episode_reward = self.run_episode(training=True)
        self.episode_count += 1

        # ── 2. Update epsilon (exploration decay) ───────────────────────────
        self.epsilon = self.compute_epsilon(self.episode_count)

        # ── 3. Sample a batch and compute loss ──────────────────────────────
        if len(self.buffer) >= self.config.batch_size:
            loss = self.compute_loss()

            # ── 4. Gradient update ──────────────────────────────────────────
            self.optimizer.zero_gradients()
            loss.backward()
            self.optimizer.step()

        # ── 5. Soft-update target networks ──────────────────────────────────
        self.soft_update_targets()

        return episode_reward

    # ─────────────────────────────────────────────────────────────────────────
    # Episode runner
    # ─────────────────────────────────────────────────────────────────────────

    def run_episode(self, training, env):
        """
        Run a complete episode in the environment.

        training=True: uses epsilon-greedy action selection, stores transitions
        training=False: uses greedy (epsilon=0) action selection, no storage
        """
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            s0, s1 = extract_per_agent_observations(state)
            z = concatenate(s0, s1)

            # Action selection (each agent independent)
            eps = self.epsilon if training else 0.0
            a0 = self.q_network_agent0.select_action(s0, epsilon=eps)
            a1 = self.q_network_agent1.select_action(s1, epsilon=eps)

            # Environment step
            next_state, env_reward, done, info = env.step([a0, a1])
            r_team = extract_team_reward(env_reward, info)

            # Store transition in replay buffer (training only)
            if training:
                s0_next, s1_next = extract_per_agent_observations(next_state)
                z_next = concatenate(s0_next, s1_next)
                self.buffer.store(s0, s1, z, a0, a1, r_team, done,
                                  s0_next, s1_next, z_next)

            episode_reward += r_team
            state = next_state

        return episode_reward

    # ─────────────────────────────────────────────────────────────────────────
    # Loss computation — the heart of QMIX
    # ─────────────────────────────────────────────────────────────────────────

    def compute_loss(self):
        """
        QMIX Bellman TD loss.

        Steps:
        1. Sample a batch of transitions from the replay buffer
        2. Compute Q_tot for current transitions using online networks
        3. Compute Q_tot_next for next states using TARGET networks (no gradient)
        4. Compute Bellman target: r + gamma * Q_tot_next * (1 - done)
        5. Compute loss between Q_tot and Bellman target
        """
        batch = self.buffer.sample(self.config.batch_size)

        # ── Step 2: Current Q_tot ────────────────────────────────────────────
        # Get Q-values for the actions actually taken (from online DQNs)
        q0 = self.q_network_agent0.get_q_for_action(batch['s0'], batch['a0'])
        q1 = self.q_network_agent1.get_q_for_action(batch['s1'], batch['a1'])
        agent_qs = concatenate_agent_values(q0, q1)  # (batch, num_agents)

        # Mix into Q_tot using online mixing network
        q_tot = self.mixing_network.forward(agent_qs, batch['z'])  # (batch,)

        # ── Step 3: Next-state Q_tot (from TARGET networks — no gradient) ────
        with no_gradient():
            # Greedy Q-values for next state from TARGET DQNs
            q0_next = self.target_q_agent0.forward(batch['s0_next'])
            q1_next = self.target_q_agent1.forward(batch['s1_next'])

            # Take greedy (argmax) Q-values for next state
            q0_next_greedy = max_over_actions(q0_next)
            q1_next_greedy = max_over_actions(q1_next)
            agent_qs_next = concatenate_agent_values(q0_next_greedy, q1_next_greedy)

            # Mix using TARGET mixing network
            q_tot_next = self.target_mixing_network.forward(agent_qs_next, batch['z_next'])

        # ── Step 4: Bellman target ───────────────────────────────────────────
        # Multiply by (1 - done) to zero out terminal transitions
        bellman_target = batch['r'] + self.config.gamma * q_tot_next * (1.0 - batch['done'])

        # ── Step 5: Loss ─────────────────────────────────────────────────────
        # Robust loss function (less sensitive to outlier Q-value errors
        # during early high-variance exploration than squared loss)
        loss = robust_regression_loss(q_tot, stop_gradient(bellman_target))

        return loss

    # ─────────────────────────────────────────────────────────────────────────
    # Target network maintenance
    # ─────────────────────────────────────────────────────────────────────────

    def soft_update_targets(self):
        """
        Apply Polyak averaging to ALL three target networks simultaneously.
        τ determines the update rate — typically a small value (~0.01–0.05).

        All three target networks (Q̄₀, Q̄₁, mixer_target) are updated together
        at the end of each training epoch.
        """
        tau = self.config.tau
        self.target_q_agent0.soft_update(self.q_network_agent0, tau)
        self.target_q_agent1.soft_update(self.q_network_agent1, tau)
        self.target_mixing_network.soft_update(self.mixing_network, tau)

    # ─────────────────────────────────────────────────────────────────────────
    # Exploration schedule
    # ─────────────────────────────────────────────────────────────────────────

    def compute_epsilon(self, episode):
        """
        Exponential decay schedule for exploration probability.

        epsilon(t) = epsilon_min + (1 - epsilon_min) * exp(-decay_rate * t)

        This is the most sensitive hyperparameter in the QMIX system.
        Too fast: agents commit to poor strategies before coordination emerges.
        Too slow: exploration noise persists long after convergence, suppressing
                  performance on easier layouts and creating misleading benchmarks.

        The decay rate should be calibrated to the HARDEST layout being trained,
        because easier layouts will converge faster and simply carry more residual
        exploration — which is then revealed as "hidden performance" at evaluation.
        """
        decay = self.config.epsilon_decay_rate
        eps_min = self.config.epsilon_min
        epsilon = eps_min + (1.0 - eps_min) * exp(-decay * episode)
        return max(epsilon, eps_min)

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(self, env, num_episodes):
        """
        Evaluate the current policy greedily over N episodes.
        No exploration, no buffer updates, no gradient computation.

        Returns average reward (soups per episode) over all evaluation episodes.

        Note: deterministic environment + deterministic policy = every episode
        is identical. For truly comparable evaluation, deterministic_mode must
        be set in the environment.
        """
        total_soups = 0
        for _ in range(num_episodes):
            episode_reward = self.run_episode(training=False, env=env)
            total_soups += count_soups(episode_reward)

        average_soups = total_soups / num_episodes
        return average_soups


# ─────────────────────────────────────────────────────────────────────────────
# Top-level training entrypoint (conceptual)
# ─────────────────────────────────────────────────────────────────────────────

def train_qmix(layout_name, config):
    """
    Complete QMIX training run for a single kitchen layout.

    layout_name: e.g., "cramped_room", "coordination_ring", "counter_circuit"
    config: hyperparameter configuration object
    """
    # Initialize environment and agent
    env = make_overcooked_env(layout_name)
    agent = QMIXAgent(config)

    # Pre-fill replay buffer with random transitions
    agent.buffer.warm_up(env, random_policy, num_steps=config.warmup_steps)

    # Training loop
    for epoch in range(config.max_episodes):
        train_reward = agent.training_step()

        # Logging
        log('episode', epoch)
        log('epsilon', agent.epsilon)
        log('train_reward', train_reward)

    # Final evaluation
    avg_soups = agent.evaluate(env, num_episodes=100)
    log('final_eval_soups_per_episode', avg_soups)
    return agent, avg_soups
