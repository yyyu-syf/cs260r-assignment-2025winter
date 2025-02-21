"""
This script reads the saved sb3 agent and save useful data into a new file that will be used
later in `agents/agent_sb3/agent.py` without stable-baselines3 dependency.
"""

import torch
from stable_baselines3.common.save_util import load_from_zip_file

if __name__ == '__main__':
    ckpt = "runs/ppo_metadrive/ppo_metadrive_2025-02-20_20-34-09_ec54bb59/models/rl_model_320000_steps.zip"
    data, params, pytorch_variables = load_from_zip_file(ckpt)

    # Just make a customized dict to store data we want to use later.
    new_data = dict(
        action_space=data["action_space"],
        observation_space=data["observation_space"],
        state_dict=params['policy']
    )
    torch.save(new_data, "example_sb3_ppo_agent.pt")
