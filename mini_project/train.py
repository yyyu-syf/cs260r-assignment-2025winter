import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

# Import the training environment.
from env import get_training_env
# Import your agent’s Policy class (adjust the import path if needed).
from agents.agent_105514254.agent import Policy

# --- Helper Functions for PPO ---
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    """
    Compute Generalized Advantage Estimation (GAE) over a rollout.
    Args:
        rewards (np.array): Rewards over the rollout.
        values (np.array): Value estimates over the rollout.
        dones (np.array): Done flags over the rollout.
        next_value (float): The bootstrap value after the rollout.
        gamma (float): Discount factor.
        lam (float): GAE lambda parameter.
    Returns:
        advantages (np.array): Advantage estimates.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages

def extract_obs(raw_obs):
    """
    If the environment follows Gymnasium API, reset/step returns a tuple (obs, info)
    or (obs, reward, terminated, truncated, info). This function extracts the observation.
    """
    # If raw_obs is a tuple and its first element is the actual observation, return that.
    if isinstance(raw_obs, tuple):
        return raw_obs[0]
    return raw_obs

def main(args):
    # Create the training environment with the specified number of maps.
    env = get_training_env({"num_scenarios": args.num_maps})
    
    # Create your agent.
    policy = Policy()  # This instantiates PPOTrainer internally.
    if args.load_ckpt:
        print(f"Loading checkpoint from {args.load_ckpt} with suffix {args.load_suffix} ...")
        policy.agent.load_w(log_dir=args.load_ckpt, suffix=args.load_suffix)
        start_update = int(args.load_suffix[6:]) if args.load_suffix else 0
    else:
        print("No checkpoint provided, starting from scratch.")
        start_update = 0
    # Training hyperparameters.
    total_steps_per_update = args.num_steps  # Number of steps to collect per update.
    num_updates = args.total_updates         # Total PPO updates.
    ppo_epochs = args.num_epochs             # PPO epochs per update.
    mini_batch_size = args.mini_batch_size   # Mini-batch size for PPO update.

    # PPO-specific parameters (read from the agent's config).
    gamma = 0.99
    lam = 0.95
    clip_param = policy.agent.config.ppo_clip_param
    value_loss_weight = policy.agent.config.value_loss_weight
    entropy_loss_weight = policy.agent.config.entropy_loss_weight
    grad_norm_max = policy.agent.grad_norm_max

    optimizer = policy.agent.optimizer

    print(f"Starting training on {args.num_maps} maps for {num_updates} updates...")
    start_time = time.time()

    for update in range(start_update,num_updates):
        # Buffers to store rollout data.
        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        dones_buffer = []
        log_probs_buffer = []
        values_buffer = []

        # Reset environment.
        raw_obs = env.reset()
        obs = extract_obs(raw_obs)
        done = False

        # Collect a rollout of fixed number of steps.
        for step in range(total_steps_per_update):
            # Compute action and value.
            value, action, log_prob = policy.agent.compute_action(obs)
            if action.ndim > 1:
                action = action[0].tolist()

            # Store current observation, action, log_prob, and value.
            obs_buffer.append(obs)
            actions_buffer.append(action)
            log_probs_buffer.append(log_prob.detach().cpu().numpy())
            values_buffer.append(value.detach().cpu().numpy().squeeze())

            # Step the environment.
            step_result = env.step(action)
            # Handle Gymnasium's API which returns (obs, reward, terminated, truncated, info)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                raw_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raw_obs, reward, done, info = step_result

            obs = extract_obs(raw_obs)
            rewards_buffer.append(reward)
            dones_buffer.append(done)

            # If episode finished, reset the environment.
            if done:
                raw_obs = env.reset()
                obs = extract_obs(raw_obs)

        # Convert rollout lists to numpy arrays.
        obs_buffer = np.array(obs_buffer)           # Shape: (T, obs_dim) or similar.
        actions_buffer = np.array(actions_buffer)     # Shape: (T, act_dim)
        rewards_buffer = np.array(rewards_buffer).squeeze()  # Shape: (T,)
        dones_buffer = np.array(dones_buffer).astype(np.float32)  # Shape: (T,)
        log_probs_buffer = np.array(log_probs_buffer).squeeze()  # Shape: (T,)
        values_buffer = np.array(values_buffer)       # Shape: (T,)

        # Bootstrap: get value for the last observation.
        obs_tensor = policy.agent.process_obs(obs)
        with torch.no_grad():
            next_value = policy.agent.compute_values(obs_tensor).cpu().numpy().squeeze()

        # Compute advantages and returns.
        advantages = compute_gae(rewards_buffer, values_buffer, dones_buffer, next_value, gamma, lam)
        returns = advantages + values_buffer

        # Convert data to torch tensors.
        device = policy.agent.device
        obs_tensor = policy.agent.process_obs(obs_buffer)  # shape: (T, obs_dim)
        actions_tensor = torch.tensor(actions_buffer, dtype=torch.float32, device=device)
        old_log_probs = torch.tensor(log_probs_buffer, dtype=torch.float32, device=device).unsqueeze(1)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device).unsqueeze(1)

        # Normalize advantages.
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update: shuffle indices and iterate in mini-batches.
        num_samples = obs_tensor.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = indices[start:end]

                mb_obs = obs_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_returns = returns_tensor[mb_inds]
                mb_advantages = advantages_tensor[mb_inds]

                # Evaluate current policy.
                new_values, new_log_probs, entropy = policy.agent.evaluate_actions(mb_obs, mb_actions)

                # Compute the ratio (pi_new / pi_old).
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss.
                value_loss = F.mse_loss(new_values, mb_returns)

                # Entropy loss.
                entropy_loss = -entropy.mean()

                # Total loss.
                loss = policy_loss + value_loss_weight * value_loss + entropy_loss_weight * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.agent.model.parameters(), grad_norm_max)
                optimizer.step()

        avg_reward = np.mean(rewards_buffer)
        print(f"Update [{update + 1}/{num_updates}] - Loss: {loss.item():.4f} - Avg Reward: {avg_reward:.2f}")

        # Save model periodically.
        if (update + 1) % policy.agent.config.save_freq == 0:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = policy.agent.save_w(log_dir="checkpoints", suffix=f"update{update + 1}")
            print(f"Saved model checkpoint to {save_path}")

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO Agent in MetaDrive Environment")
    parser.add_argument("--num_maps", type=int, default=3,
                        help="Number of maps (num_scenarios) for training (e.g., 1, 3, or 10)")
    parser.add_argument("--num_steps", type=int, default=4000,
                        help="Number of steps per rollout update")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of PPO epochs per update")
    parser.add_argument("--mini_batch_size", type=int, default=256,
                        help="Mini-batch size for PPO updates")
    parser.add_argument("--total_updates", type=int, default=1000,
                        help="Total number of PPO updates to perform")
    parser.add_argument("--load_ckpt", type=str, default="",
                    help="Directory of the checkpoint to load and resume training from")
    parser.add_argument("--load_suffix", type=str, default="",
                    help="Suffix for the checkpoint file to load (e.g., 'update100')")
    args = parser.parse_args()
    main(args)