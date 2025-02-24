import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --------------------------
# Utility: Extract observation from env.reset()/step()
# --------------------------
def extract_obs(o):
    if isinstance(o, tuple):
        return o[0]
    return o

# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# --------------------------
# Neural Network Helper
# --------------------------
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# --------------------------
# Actor Network
# --------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return y_t, log_prob

# --------------------------
# Critic Network
# --------------------------
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# --------------------------
# SAC Agent with Checkpoint Support
# --------------------------
class SACAgent:
    def __init__(self, state_dim, action_dim, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=-1.0, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = target_entropy

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )

        self.critic1_target = Critic(state_dim, action_dim).to(self.device)
        self.critic2_target = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor.forward(state)
            action = torch.tanh(mean)
            return action.cpu().detach().numpy()[0]
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().detach().numpy()[0]

    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic1_target(next_state, next_action)
            target_q2 = self.critic2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_value = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic_loss = nn.MSELoss()(current_q1, target_value) + nn.MSELoss()(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()

        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename, step):
        torch.save({
            'step': step,
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
        }, "checkpoints/1/"+filename)
        print(f"Checkpoint saved at step {step} to {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.log_alpha = checkpoint['log_alpha']
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        saved_step = checkpoint.get('step', 0)
        print(f"Loaded checkpoint from {filename} at step {saved_step}")
        return saved_step

# --------------------------
# Training Loop with Checkpointing
# --------------------------
def train_agent(env, agent, total_steps=200_000, batch_size=256, start_steps=1000, update_after=1000, update_every=50, save_every=10000, resume_step=0):
    replay_buffer = ReplayBuffer(capacity=1_000_000)
    state = extract_obs(env.reset())
    episode_reward = 0
    episode_rewards = []
    current_step = resume_step

    while current_step < total_steps:
        current_step += 1
        if current_step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_raw = env.step(action)
        if isinstance(next_raw, tuple) and len(next_raw) >= 4:
            next_state, reward, done, info = next_raw[:4]
        else:
            next_state, reward, done, info = next_raw, 0.0, False, {}
        next_state = extract_obs(next_state)

        replay_buffer.push(state, action, reward, next_state, float(done))
        state = next_state
        episode_reward += reward

        if done:
            state = extract_obs(env.reset())
            episode_rewards.append(episode_reward)
            episode_reward = 0

        if current_step >= update_after and current_step % update_every == 0:
            for _ in range(update_every):
                agent.update(replay_buffer, batch_size)

        if current_step % save_every == 0:
            agent.save(f"ckpt_{current_step}.pth", current_step)
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Step {current_step}, Average Reward (last 10 episodes): {avg_reward:.2f}")

    return episode_rewards

# --------------------------
# Main: Argument Parsing and Training
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent on MetaDrive Safety Environment")
    parser.add_argument("--num_maps", type=int, default=3, help="Number of training maps (e.g., 1, 3, or 10)")
    parser.add_argument("--total_steps", type=int, default=200000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for updates")
    parser.add_argument("--save_every", type=int, default=10000, help="Save checkpoint every N steps")
    parser.add_argument("--load_ckpt", type=str, default="", help="Path to checkpoint file to resume from")
    args = parser.parse_args()

    # Import MetaDrive training environment
    from env import get_training_env
    env = get_training_env({"num_scenarios": args.num_maps})

    # Define state and action dimensions (adjust based on your environment)
    state_dim = 259
    action_dim = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(state_dim, action_dim, device=device)

    # If a checkpoint path is provided, load it and resume training
    resume_step = 0
    if args.load_ckpt != "":
        resume_step = agent.load(args.load_ckpt)

    # Train the agent starting from resume_step
    rewards = train_agent(env, agent, total_steps=args.total_steps, batch_size=args.batch_size, save_every=args.save_every, resume_step=resume_step)

    # Final save
    agent.save("final_sac_metadrive_model.pth", args.total_steps)
    print("Training complete. Final model saved as final_sac_metadrive_model.pth")