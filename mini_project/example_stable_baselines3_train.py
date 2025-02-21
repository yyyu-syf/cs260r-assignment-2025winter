"""
Install dependencies:
    pip install stable-baselines3[extra]
    pip install wandb

Usage:
    python example_stable_baselines3.py
"""
import argparse
import datetime
import logging
import os
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
from metadrive.engine.logger import set_log_level
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from stable_baselines3.ppo.policies import ActorCriticPolicy
from wandb.integration.sb3 import WandbCallback

import wandb
from env import get_training_env, get_validation_env

# Remove MetaDrive's logging information when episode ends.
set_log_level(logging.ERROR)


def get_time_str():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def remove_reset_seed_and_add_monitor(make_env, trial_dir):
    """
    MetaDrive env's reset function takes a seed argument and use it to determine the map to load.
    However, in stable-baselines3, it calls reset function with a seed argument serving as the random seed,
    which is not what we want. We do a trick here to remap the random seed to map index.

    Stable-baselines3 recommends using Monitor wrapper to log training data. We add a Monitor wrapper here.
    """
    from gymnasium import Wrapper
    from stable_baselines3.common.monitor import Monitor
    class NewClass(Wrapper):
        def reset(self, seed=None, **kwargs):
            # PZH: We do a trick here to remap the seed to the map index. This can help randomize the maps.
            if seed is not None:
                new_seed = self.env.start_index + (seed % self.env.num_scenarios)
            else:
                new_seed = None
            return self.env.reset(seed=new_seed, **kwargs)

    def new_make_env():
        env = make_env()
        NewClass.__name__ = env.__class__.__name__ + "WithoutResetSeed"
        wrapped_env = NewClass(env)
        wrapped_env = Monitor(env=wrapped_env, filename=str(trial_dir))
        return wrapped_env

    return new_make_env


class CustomizedEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluations_info_buffer = defaultdict(list)

    def _log_success_callback(self, locals_, globals_):
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

            maybe_is_success2 = info.get("arrive_dest", None)
            if maybe_is_success2 is not None:
                self._is_success_buffer.append(maybe_is_success2)

            assert (maybe_is_success is None) or (maybe_is_success2 is None), "We cannot have two success flags!"

            for k in ["route_completion", "total_cost", "arrive_dest", "max_step", "out_of_road", "crash"]:
                if k in info:
                    self.evaluations_info_buffer[k].append(info[k])

        if "raw_action" in info:
            self.evaluations_info_buffer["raw_action"].append(info["raw_action"])

    def _on_step(self) -> bool:
        """
        PZH Note: Overall this function is copied from original EvalCallback._on_step.
        We additionally record evaluations_info_buffer to the logger.
        """

        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.vec_env import sync_envs_normalization

        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                # PZH: Save evaluations_info_buffer to the log file
                for k, v in self.evaluations_info_buffer.items():
                    kwargs[k] = v

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,  # type: ignore[arg-type]
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # PZH: Add this metric.
            self.logger.record("eval/num_episodes", len(episode_rewards))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # PZH: We record evaluations_info_buffer to the logger
            for k, v in self.evaluations_info_buffer.items():
                self.logger.record("eval/{}".format(k), np.mean(np.asarray(v)))

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="ppo_metadrive", type=str, help="The name for this batch of experiments.")
    parser.add_argument("--ckpt", default=None, type=str, help="Path to previous checkpoint.")
    parser.add_argument("--wandb", action="store_true", help="Set to True to upload stats to wandb.")
    args = parser.parse_args()

    # ===== Set up some arguments =====
    experiment_batch_name = "{}".format(args.exp_name)
    trial_name = "{}_{}_{}".format(experiment_batch_name, get_time_str(), uuid.uuid4().hex[:8])
    use_wandb = args.wandb
    experiment_dir = Path("runs") / experiment_batch_name
    trial_dir = experiment_dir / trial_name
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(trial_dir, exist_ok=True)
    print(f"We start logging training data into {trial_dir}")

    # ===== Setup environment =====
    num_train_envs = 32
    num_eval_envs = 5
    train_env = make_vec_env(remove_reset_seed_and_add_monitor(get_training_env, trial_dir), n_envs=num_train_envs,
                             vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(remove_reset_seed_and_add_monitor(get_validation_env, trial_dir), n_envs=num_eval_envs,
                            vec_env_cls=SubprocVecEnv)

    # ===== Setup the callbacks =====
    save_freq = 10_000  # Number of steps per model checkpoint
    eval_freq = 100_000  # Number of steps per evaluation
    wandb_save_freq = 100_000  # Number of steps per evaluation
    num_eval_episodes = 50
    checkpoint_callback = CheckpointCallback(
        name_prefix="rl_model",
        verbose=2,
        save_freq=save_freq,
        save_path=str(trial_dir / "models")
    )
    eval_callback = CustomizedEvalCallback(
        eval_env,
        best_model_save_path=str(trial_dir / "eval"),
        log_path=str(trial_dir / "eval"),
        eval_freq=max(eval_freq // num_train_envs, 1),
        n_eval_episodes=num_eval_episodes,
    )
    callbacks = [checkpoint_callback, eval_callback]
    if use_wandb:
        wandb.init(
            project="cs260r",
            id=trial_name,
            name=experiment_batch_name,
            sync_tensorboard=True,
            dir=str(trial_dir),
        )
        callbacks.append(WandbCallback(model_save_path=str(trial_dir / "wandb_models"), model_save_freq=wandb_save_freq))
    callbacks = CallbackList(callbacks)

    # ===== Setup the training algorithm =====
    model = PPO(
        env=train_env,
        policy=ActorCriticPolicy,
        n_steps=512,  # n_steps * n_envs = total_batch_size
        n_epochs=20,
        learning_rate=5e-5,
        batch_size=256,
        clip_range=0.1,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=10.0,
        tensorboard_log=str(trial_dir),
        verbose=2,
        device="auto",
    )

    if args.ckpt:
        ckpt = Path(args.ckpt)
        print(f"Loading checkpoint from {ckpt}!")
        from stable_baselines3.common.save_util import load_from_zip_file
        data, params, pytorch_variables = load_from_zip_file(ckpt, device=model.device, print_system_info=False)
        model.set_parameters(params, exact_match=True, device=model.device)

    # ===== Launch training =====
    total_timesteps = 1_000_000  # 1M steps
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
        tb_log_name=experiment_batch_name,
        log_interval=1,
        progress_bar=True,
    )
