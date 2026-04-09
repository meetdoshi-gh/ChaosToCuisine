"""
replay_buffer.py — Experience Replay Memory
============================================
Pseudocode level: 4/10 (conceptual structure, no implementation details)

The replay buffer stores past interaction transitions and allows the
training algorithm to sample them randomly later.

Why experience replay matters for QMIX:
1. Breaks temporal correlations between consecutive transitions
   (which would bias gradient estimates if trained sequentially)
2. Enables off-policy learning — the agent can learn from transitions
   collected under different policies (including older, more exploratory ones)
3. Allows the same transition to be used in multiple gradient updates,
   improving sample efficiency

The buffer operates as a fixed-capacity circular queue:
- New transitions overwrite the oldest ones once capacity is reached
- Sampling is uniform random — every stored transition has equal probability

QMIX Transition Format:
    (s₀, s₁, z, a₀, a₁, r_team, done, s₀', s₁', z')
    where z = [s₀; s₁] and z' = [s₀'; s₁'] (joint states)
"""


class ReplayBuffer:
    """
    Fixed-capacity circular buffer for QMIX transitions.

    Stores:
        - Per-agent local observations (s₀, s₁) — needed for individual DQNs
        - Joint centralized state z = [s₀; s₁] — needed for hypernetworks/mixer
        - Actions taken by each agent (a₀, a₁)
        - Shared team reward r_team
        - Episode termination flag done
        - Next-step observations and joint state (s₀', s₁', z')
    """

    def __init__(self, capacity):
        """
        capacity: maximum number of transitions to store
        Older transitions are overwritten once capacity is reached.
        """
        self.capacity = capacity
        self.buffer = []
        self.write_position = 0  # Circular index

    def store(self, observation_agent0, observation_agent1, joint_state,
              action_agent0, action_agent1, team_reward, done,
              next_obs_agent0, next_obs_agent1, next_joint_state):
        """
        Add one transition to the buffer.

        All inputs are for a single timestep in a single episode.
        The buffer does NOT require complete episodes — individual step
        transitions are stored independently.
        """
        transition = {
            's0': observation_agent0,        # Agent 0 local obs
            's1': observation_agent1,        # Agent 1 local obs
            'z':  joint_state,               # [s₀; s₁] centralized
            'a0': action_agent0,             # Discrete action index
            'a1': action_agent1,
            'r':  team_reward,               # Shared team reward for this step
            'done': done,                    # 1.0 if terminal, 0.0 otherwise
            's0_next': next_obs_agent0,
            's1_next': next_obs_agent1,
            'z_next': next_joint_state,
        }

        if len(self.buffer) < self.capacity:
            # Buffer not yet full — append
            self.buffer.append(transition)
        else:
            # Buffer full — overwrite oldest transition (circular)
            self.buffer[self.write_position] = transition

        self.write_position = (self.write_position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a random batch of transitions for training.

        batch_size: number of transitions to return
        returns: dictionary of batched tensors ready for training

        Sampling is uniform — all stored transitions equally likely.
        This is important: prioritized sampling (PER) would improve
        credit assignment efficiency but is not implemented here.
        """
        assert len(self.buffer) >= batch_size, "Not enough transitions to sample"

        # Sample random indices without replacement
        indices = random_sample_without_replacement(len(self.buffer), batch_size)
        sampled = [self.buffer[i] for i in indices]

        # Stack individual fields into batched tensors
        batch = {
            's0':     stack([t['s0'] for t in sampled]),
            's1':     stack([t['s1'] for t in sampled]),
            'z':      stack([t['z'] for t in sampled]),
            'a0':     stack([t['a0'] for t in sampled]),
            'a1':     stack([t['a1'] for t in sampled]),
            'r':      stack([t['r'] for t in sampled]),
            'done':   stack([t['done'] for t in sampled]),
            's0_next': stack([t['s0_next'] for t in sampled]),
            's1_next': stack([t['s1_next'] for t in sampled]),
            'z_next': stack([t['z_next'] for t in sampled]),
        }
        return batch

    def warm_up(self, env, random_policy, num_steps):
        """
        Pre-fill buffer with random-action transitions before training begins.

        Why: if training starts immediately, the first batches contain only
        highly correlated transitions from the agent's initial random-ish policy.
        Pre-filling with deliberate random-action episodes gives a more
        diverse baseline of experience to start learning from.

        env: the Overcooked gym environment
        random_policy: function that returns random actions for both agents
        num_steps: number of transitions to collect before training starts
        """
        state = env.reset()
        for _ in range(num_steps):
            a0, a1 = random_policy()
            next_state, reward, done, info = env.step([a0, a1])

            s0, s1 = extract_per_agent_observations(state)
            z = concatenate(s0, s1)
            s0_next, s1_next = extract_per_agent_observations(next_state)
            z_next = concatenate(s0_next, s1_next)
            r_team = extract_team_reward(reward, info)

            self.store(s0, s1, z, a0, a1, r_team, done,
                       s0_next, s1_next, z_next)

            if done:
                state = env.reset()
            else:
                state = next_state

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self, min_size):
        """True if buffer has enough transitions to start sampling batches."""
        return len(self.buffer) >= min_size
