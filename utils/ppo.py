from collections import defaultdict
import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange
import gc


def preprocess_dataframe(df: pd.DataFrame):
    """
    Build a station-indexed adjacency structure from a contextual edge DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: ["from", "to", "departure_time", "duration", "type"].
        - "from"/"to": station identifiers (strings)
        - "departure_time": minutes from 0..1440 (NaN for transfer edges)
        - "duration": hop duration in minutes
        - "type": {"trip", "transfer"}

    Returns
    -------
    graph : dict[int, list[dict]]
        For each station index, a list of outgoing edges dictionaries:
        {"to": int, "departure_time": float|None, "duration": float, "type": str}.
    station_to_idx : dict[str, int]
        Mapping from original station id to compact index [0..num_stations-1].
    idx_to_station : dict[int, str]
        Inverse mapping from index to original station id.
    num_stations : int
        Number of unique stations.

    Notes
    -----
    Departure times for transfers are kept as None in the graph and will be
    replaced at observation time with the current simulation time.
    """
    df = df.copy().reset_index(drop=True)

    # Build station index
    stations = pd.unique(df[["from", "to"]].values.ravel())
    station_to_idx = {s: i for i, s in enumerate(stations)}
    idx_to_station = {i: s for s, i in station_to_idx.items()}
    num_stations = len(station_to_idx)

    # Build adjacency-like structure per station
    graph = defaultdict(list)
    for _, row in df.iterrows():
        from_id = station_to_idx[row["from"]]
        to_id = station_to_idx[row["to"]]
        graph[from_id].append(
            {
                "to": to_id,
                "departure_time": (
                    None
                    if pd.isna(row["departure_time"])
                    else float(row["departure_time"])
                ),
                "duration": float(row["duration"]),
                "type": str(row["type"]),
            }
        )
    return graph, station_to_idx, idx_to_station, num_stations


class SafeEnv(gym.Env):
    """
    Wrapper that shields against hard failures in reset/step and returns dummy fallbacks.

    This is useful when parallel environments are used and a single env crash
    would otherwise stop the entire rollout.

    Parameters
    ----------
    env : gym.Env
        The underlying environment to wrap.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, **kwargs):
        """Reset the environment with error handling. Returns (obs, info)."""
        try:
            return self.env.reset(**kwargs)
        except Exception as e:
            print(f"[SAFE RESET ERROR] {e}")
            dummy_obs = self.observation_space.sample()
            # Flag the observation as invalid so the trainer can ignore it
            if isinstance(dummy_obs, dict):
                dummy_obs["invalid"] = True
            return dummy_obs, {}

    def step(self, action):
        """Step the environment with error handling. Returns (obs, reward, terminated, truncated, info)."""
        try:
            return self.env.step(action)
        except Exception as e:
            print(f"[SAFE STEP ERROR] {e}")
            dummy_obs = self.observation_space.sample()
            if isinstance(dummy_obs, dict):
                dummy_obs["invalid"] = True
            # Penalize and terminate to prevent getting stuck
            return dummy_obs, -10.0, True, True, {}

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()


# --------- DATA-BASED TRANSPORT ENVIRONMENT ---------
class TransportEnvDF(gym.Env):
    """
    Time-dependent transport environment backed by a precomputed edge table.

    The agent starts at a station and a time-of-day and must reach a given
    destination by selecting among a limited set of feasible outgoing options.
    Trip edges are only available if their departure_time >= current_time;
    transfer edges are always available at the current time.

    Observation space
    -----------------
    Dict({
        "current_station": Discrete(num_stations),
        "destination": Discrete(num_stations),
        "hour": Box(shape=(2,), low=0, high=1)  # [sin(t), cos(t)] encoding of current time
        "options": Box(shape=(max_actions, 4))  # rows: [to_norm, dep_time_norm, duration_norm, valid]
    })

    Action space
    ------------
    Discrete(max_actions)  # index in the options matrix

    Parameters
    ----------
    graph : dict[int, list[dict]]
        Output of `preprocess_dataframe`, adjacency-like structure.
    station_to_idx : dict[str, int]
        Station id to index mapping.
    idx_to_station : dict[int, str]
        Index to station id mapping.
    num_stations : int
        Number of stations.
    max_actions : int, default=5
        How many options to expose per state (truncated list of valid edges).
    max_steps : int, default=20
        Maximum steps before truncation.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        graph: dict[int, list[dict]],
        station_to_idx: dict[int, list[dict]],
        idx_to_station: dict[int, str],
        num_stations: int,
        max_actions: int = 5,
        max_steps: int = 20,
    ):
        super().__init__()
        self.graph = graph
        self.station_to_idx = station_to_idx
        self.idx_to_station = idx_to_station
        self.num_stations = num_stations
        self.max_actions = max_actions
        self.max_steps = max_steps

        self.observation_space = spaces.Dict(
            {
                "current_station": spaces.Discrete(self.num_stations),
                "destination": spaces.Discrete(self.num_stations),
                "hour": spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                "options": spaces.Box(
                    low=0, high=1, shape=(max_actions, 4), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(max_actions)

    def reset(self, seed=None, options=None):
        """
        Reset environment state.

        Parameters
        ----------
        seed : int | None
            Optional seed used by Gym to initialize self.np_random.
        options : dict | None
            If {'easy': True}, the destination is sampled among the neighbors
            of the starting station to simplify exploration.

        Returns
        -------
        (obs, info) : tuple
            Observation dict and empty info.
        """
        super().reset(seed=seed)

        easy = options.get("easy", False) if options else False

        self.current_station = int(self.np_random.integers(self.num_stations))
        if easy:
            neighbors = [e["to"] for e in self.graph[self.current_station]]
            if not neighbors:
                raise ValueError(
                    f"[RESET ERROR] No neighbors for station {self.current_station}"
                )
            self.destination = int(self.np_random.choice(neighbors))
        else:
            self.destination = int(self.np_random.integers(self.num_stations))
            while self.destination == self.current_station:
                self.destination = int(self.np_random.integers(self.num_stations))

        self.current_hour = float(self.np_random.uniform(0, 1440))  # minutes in day
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """Assemble the observation dict, filtering feasible options at current time."""
        # Time-of-day encoding
        hour_sin = np.sin(2 * np.pi * self.current_hour / 1440)
        hour_cos = np.cos(2 * np.pi * self.current_hour / 1440)

        # Build options matrix [max_actions, 4]
        options = np.zeros((self.max_actions, 4), dtype=np.float32)
        valid_options = []

        for edge in self.graph[self.current_station]:
            if edge["type"] == "trip":
                # Only upcoming trips are valid
                if (
                    edge["departure_time"] is not None
                    and edge["departure_time"] >= self.current_hour
                ):
                    valid_options.append(
                        {
                            "to": edge["to"],
                            "departure_time": edge["departure_time"],
                            "duration": edge["duration"],
                            "valid": 1.0,
                        }
                    )
            else:  # transfer: always available now
                valid_options.append(
                    {
                        "to": edge["to"],
                        "departure_time": self.current_hour,
                        "duration": edge["duration"],
                        "valid": 1.0,
                    }
                )

        # Normalize and pack (destination index, departure time, duration)
        for i, opt in enumerate(valid_options[: self.max_actions]):
            options[i] = [
                opt["to"] / self.num_stations,  # normalize station id
                opt["departure_time"] / 1440.0,  # normalize to [0,1]
                opt["duration"] / 60.0,  # minutes -> hours norm (approx)
                opt["valid"],
            ]

        return {
            "current_station": self.current_station,
            "destination": self.destination,
            "hour": np.array([hour_sin, hour_cos], dtype=np.float32),
            "options": options,
        }

    def step(self, action):
        """
        Apply the selected option and advance simulation time.

        Rewards
        -------
        - Small step penalty (-0.1) to encourage shorter paths
        - Invalid action: -1.0
        - Moving to a new station: +0.2
        - Goal reached: +100.0
        - Timeout without reaching goal: -5.0

        Returns
        -------
        (obs, reward, terminated, truncated, info)
        """

        self.steps += 1
        done = False
        reward = -0.1  # step penalty

        obs = self._get_obs()

        # Validate action bounds
        if action < 0 or action >= self.max_actions:
            valid = False
            reward = -1.0
        else:
            option = obs["options"][action]
            valid = bool(option[-1] > 0.5)

            if valid:
                # Denormalize
                next_station = int(option[0] * self.num_stations)
                departure_time = float(option[1] * 1440.0)
                duration = float(option[2] * 60.0)

                # Reward for moving
                if next_station != self.current_station:
                    reward += 0.2

                # Earliest-departure constraint for trips, immediate for transfers
                wait = max(0.0, departure_time - self.current_hour)
                self.current_hour = (self.current_hour + wait + duration) % 1440.0
                self.current_station = next_station

                # Success condition
                if self.current_station == self.destination:
                    reward = 100.0
                    done = True
            else:
                reward = -1.0

        # Time/step limit
        if self.steps >= self.max_steps:
            done = True
            if self.current_station != self.destination:
                reward = -5.0

        terminated = done and (self.current_station == self.destination)
        truncated = done and (self.current_station != self.destination)

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        """Print a user-friendly snapshot of the current state."""
        station_name = self.idx_to_station[self.current_station]
        destination_name = self.idx_to_station[self.destination]
        print(
            f"At station {station_name}, aiming for {destination_name}, hour: {self.current_hour:.2f}"
        )


def make_env_fn(
    graph, station_to_idx, idx_to_station, num_stations, max_actions=5, max_steps=20
):
    """
    Factory that builds a SafeEnv-wrapped TransportEnvDF (compatible with vectorized envs).

    Returns
    -------
    Callable[[], gym.Env]
        A thunk that returns a configured SafeEnv(TransportEnvDF(...)).
    """

    def _init():
        try:
            env = TransportEnvDF(
                graph,
                station_to_idx,
                idx_to_station,
                num_stations,
                max_actions=max_actions,
                max_steps=max_steps,
            )
            return SafeEnv(env)
        except Exception as e:
            print(f"[INIT ERROR] {e}")
            raise

    return _init


# --------- POLICY NETWORK ---------
class PolicyNetwork(nn.Module):
    """
    Policy that scores each presented option and outputs logits over actions.

    Parameters
    ----------
    num_stations : int
        Vocabulary size for station embeddings.
    option_dim : int, default=3
        Number of continuous fields per option row AFTER the 'to' id
        (i.e., [dep_time_norm, duration_norm, valid]).
    embed_dim : int, default=16
        Dimension of station embeddings.
    hidden_dim : int, default=64
        Hidden size in the option encoder MLP.
    """

    def __init__(
        self,
        num_stations: int,
        option_dim: int = 3,
        embed_dim: int = 16,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.station_embed = nn.Embedding(num_stations, embed_dim)

        # Each option is embedded by concatenating the embedded destination station
        # with the option continuous features, then scored by a small MLP.
        self.option_encoder = nn.Sequential(
            nn.Linear(embed_dim + option_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_station: torch.Tensor,
        destination: torch.Tensor,
        hour: torch.Tensor,
        options: torch.Tensor,
    ):
        """
        Compute unnormalized scores (logits) for each option.

        Parameters
        ----------
        current_station : torch.Tensor [B]
        destination : torch.Tensor [B]
        hour : torch.Tensor [B, 2]
            Sin/cos time-of-day features.
        options : torch.Tensor [B, max_actions, 4]
            Normalized option rows: [to_norm, dep_time_norm, duration_norm, valid].

        Returns
        -------
        logits : torch.Tensor [B, max_actions]
            One score per option (no softmax applied).
        """
        # Global context (not directly used in this simple option scorer, but can be concatenated later)
        pos_emb = self.station_embed(current_station)
        dest_emb = self.station_embed(destination)
        state = torch.cat([pos_emb, dest_emb, hour], dim=-1)  # available for extensions

        # Embed the destination station for each option row
        station_ids = (
            (options[:, :, 0] * self.station_embed.num_embeddings)
            .long()
            .clamp_(0, self.station_embed.num_embeddings - 1)
        )
        option_embs = self.station_embed(station_ids)

        # Keep only the continuous fields (dep_time, duration, valid)
        option_inputs = torch.cat([option_embs, options[:, :, 1:]], dim=-1)

        logits = self.option_encoder(option_inputs).squeeze(-1)
        return logits


# --------- CRITIC NETWORK ---------
class CriticNetwork(nn.Module):
    """
    State-value function approximator V(s).

    Parameters
    ----------
    num_stations : int
        Vocabulary size for station embeddings.
    embed_dim : int, default=16
        Dimension of station embeddings.
    hidden_dim : int, default=64
        Hidden size of the MLP head.
    """

    def __init__(self, num_stations: int, embed_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.station_embed = nn.Embedding(num_stations, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(
                embed_dim * 2 + 2, hidden_dim
            ),  # [current_emb, dest_emb, hour_sin, hour_cos]
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        current_station: torch.Tensor,
        destination: torch.Tensor,
        hour: torch.Tensor,
    ):
        """
        Compute scalar value estimate for the current state.

        Parameters
        ----------
        current_station : torch.Tensor [B]
        destination : torch.Tensor [B]
        hour : torch.Tensor [B, 2]

        Returns
        -------
        value : torch.Tensor [B]
            State-value estimate.
        """
        pos_emb = self.station_embed(current_station)
        dest_emb = self.station_embed(destination)
        x = torch.cat([pos_emb, dest_emb, hour], dim=-1)
        value = self.fc(x)
        return value.squeeze(-1)


# --------- RUNNER UTILS ---------
def select_action(policy_net: PolicyNetwork, obs: dict):
    """
    Sample an action from the policy given a single observation (numpy dict).

    Parameters
    ----------
    policy_net : PolicyNetwork
        Trained (or training) policy network.
    obs : dict
        Environment observation with numpy arrays.

    Returns
    -------
    action : int
        Selected action index in [0..max_actions-1].
    log_prob : torch.Tensor
        Log-probability of the sampled action (for PPO).
    """
    current_station = torch.tensor([obs["current_station"]])
    destination = torch.tensor([obs["destination"]])
    hour = torch.tensor(obs["hour"]).unsqueeze(0)
    options = torch.tensor(obs["options"]).unsqueeze(0)

    logits = policy_net(current_station, destination, hour, options)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob


def evaluate_action(policy_net: PolicyNetwork, obs: dict, action: int):
    """
    Compute log-probability and entropy for a given (obs, action) pair.

    Parameters
    ----------
    policy_net : PolicyNetwork
    obs : dict
        Observation dict (numpy).
    action : int
        Action index.

    Returns
    -------
    log_prob : torch.Tensor
        Log-probability of the provided action.
    entropy : torch.Tensor
        Categorical entropy for exploration bonus.
    """
    current_station = torch.tensor([obs["current_station"]])
    destination = torch.tensor([obs["destination"]])
    hour = torch.tensor(obs["hour"]).unsqueeze(0)
    options = torch.tensor(obs["options"]).unsqueeze(0)

    logits = policy_net(current_station, destination, hour, options)
    dist = Categorical(logits=logits)
    log_prob = dist.log_prob(torch.tensor([action]))
    entropy = dist.entropy()
    return log_prob, entropy


def evaluate_value(critic_net: CriticNetwork, obs: dict):
    """
    Compute V(s) for a given observation.

    Parameters
    ----------
    critic_net : CriticNetwork
    obs : dict
        Observation dict (numpy arrays).

    Returns
    -------
    value : torch.Tensor
        Scalar value estimate for the state.
    """
    current_station = torch.tensor([obs["current_station"]])
    destination = torch.tensor([obs["destination"]])
    hour = torch.tensor(obs["hour"]).unsqueeze(0)
    return critic_net(current_station, destination, hour)


# --------- PPO TRAINING LOOP ---------


def ppo_train(
    envs: gym.vector.VectorEnv | gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv,
    policy_net: PolicyNetwork,
    critic_net: CriticNetwork,
    optimizer_policy: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer,
    num_episodes: int = 500,
    gamma: float = 0.99,
    clip_eps: float = 0.2,
    critic_coef: float = 0.5,
    entropy_coef: float = 0.01,
    device: str = "cpu",
):
    """
    Proximal Policy Optimization training loop over vectorized environments.

    Parameters
    ----------
    envs : gym.vector.VectorEnv | gym.vector.SyncVectorEnv | gym.vector.AsyncVectorEnv
        Vectorized environment wrapper exposing .reset() and .step().
    policy_net : PolicyNetwork
        Policy network returning logits over actions.
    critic_net : CriticNetwork
        Value network returning V(s).
    optimizer_policy : torch.optim.Optimizer
        Optimizer for the policy network.
    optimizer_critic : torch.optim.Optimizer
        Optimizer for the critic network.
    num_episodes : int, default=500
        Number of episodes (outer rollouts).
    gamma : float, default=0.99
        Discount factor.
    clip_eps : float, default=0.2
        PPO clipping parameter.
    critic_coef : float, default=0.5
        Weight for the critic loss.
    entropy_coef : float, default=0.01
        Weight for the policy entropy bonus.
    device : str, default='cpu'
        Torch device.

    Returns
    -------
    reward_history : list[float]
        Average reward per episode across environments (for monitoring).
    """
    num_envs = envs.num_envs
    policy_net = policy_net.to(device)
    critic_net = critic_net.to(device)
    reward_history = []

    # Initial reset (warm-up)
    obs, _ = envs.reset()

    for episode in trange(num_episodes, desc="PPO training", ncols=100):
        # (Optional) Curriculum: start easy to bootstrap exploration
        easy_mode = True  # or: episode < 150
        obs, _ = envs.reset(options={"easy": easy_mode})

        log_probs = []
        values_for_policy = []
        values_for_critic = []
        rewards = []
        entropies = []
        states = []
        actions = []

        dones = [False] * num_envs

        while not all(dones):
            # Convert numpy obs to tensors on device
            current_station = torch.tensor(obs["current_station"], device=device)
            destination = torch.tensor(obs["destination"], device=device)
            hour = torch.tensor(obs["hour"], dtype=torch.float32, device=device)
            options = torch.tensor(obs["options"], dtype=torch.float32, device=device)

            # Policy & value
            logits = policy_net(current_station, destination, hour, options)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = critic_net(current_station, destination, hour)

            # Step envs (Gym >=0.26 API)
            next_obs, reward, terminated, truncated, info = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            # Skip invalid states (if SafeEnv injected one)
            if not (isinstance(obs, dict) and obs.get("invalid", False)):
                log_probs.append(log_prob)
                values_for_policy.append(value.detach())
                values_for_critic.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
                states.append(obs)
                actions.append(action)
                entropies.append(dist.entropy())
            else:
                print("[SKIP] Invalid obs detected — excluded from update.")

            obs = next_obs
            dones = done

        # Compute discounted returns
        returns = []
        R = torch.zeros(num_envs, device=device)
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns).detach()

        # Policy advantage
        log_probs = torch.stack(log_probs)
        values_policy = torch.stack(values_for_policy)
        advantages = returns - values_policy

        # Multiple policy epochs over the same trajectories
        for _ in range(4):
            new_log_probs = []
            new_entropies = []

            for s, a in zip(states, actions):
                cs = torch.tensor(s["current_station"], device=device)
                ds = torch.tensor(s["destination"], device=device)
                hr = torch.tensor(s["hour"], dtype=torch.float32, device=device)
                op = torch.tensor(s["options"], dtype=torch.float32, device=device)

                logits = policy_net(cs, ds, hr, op)
                dist = Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(a))
                new_entropies.append(dist.entropy())

            new_log_probs = torch.stack(new_log_probs)
            new_entropies = torch.stack(new_entropies)
            ratio = torch.exp(new_log_probs - log_probs.detach())

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -new_entropies.mean()

            optimizer_policy.zero_grad()
            (policy_loss + entropy_coef * entropy_loss).backward()
            optimizer_policy.step()

        # Critic update
        values_critic = torch.stack(values_for_critic)
        critic_loss = F.mse_loss(values_critic, returns)
        optimizer_critic.zero_grad()
        (critic_coef * critic_loss).backward()
        optimizer_critic.step()

        # Simple logging
        total_reward = sum([r.sum().item() for r in rewards]) / num_envs
        reward_history.append(total_reward)

        gc.collect()

        if episode % 20 == 0:
            print(
                f"\nEpisode {episode} | Avg reward: {total_reward:.2f} | Critic loss: {critic_loss.item():.4f}"
            )

    return reward_history
