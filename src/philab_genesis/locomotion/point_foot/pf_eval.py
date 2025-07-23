import argparse
import os
import torch
import glob
import re
import genesis as gs
from philab_genesis.locomotion.point_foot.pf_env import PfEnv
from philab_genesis.locomotion.point_foot.pf_config import get_cfgs, get_train_cfg
from philab_genesis.rsl_rl.runners import OnPolicyRunner
from philab_genesis.utils.visualize import Visualizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="pointfoot")
    parser.add_argument("-c", "--ckpt", type=int, default=None)
    parser.add_argument("-p", "--enable_plot", action="store_true")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, 100)
    # reward_cfg["reward_scales"] = {}

    # visualize the target
    env_cfg["visualize_target"] = True
    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the episode length to 10 seconds
    env_cfg["episode_length_s"] = 10.0

    env = PfEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    if args.ckpt is None:
        model_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoints found in {log_dir}")

        iterations = []
        for file in model_files:
            match = re.search(r"model_(\d+)\.pt", os.path.basename(file))
            if match:
                iterations.append(int(match.group(1)))

        if not iterations:
            raise ValueError(f"Could not parse iteration numbers from model files in {log_dir}")

        latest_ckpt = max(iterations)
        print(f"Using latest checkpoint: {latest_ckpt}")
    else:
        latest_ckpt = args.ckpt
        print(f"Using specified checkpoint: {latest_ckpt}")

    resume_path = os.path.join(log_dir, f"model_{latest_ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    if args.enable_plot:
        reward_names = list(env.reward_functions.keys())
        visualizer = Visualizer(reward_names)

        obs, _ = env.reset()
        with torch.no_grad():
            while True:
                actions = policy(obs)
                env.commands[0] = torch.tensor([0.5, 0.0, 0.0], device=actions.device)
                obs, _, rews, dones, infos = env.step(actions)

                reward_values = {}
                for name, reward_func in env.reward_functions.items():
                    raw_value = reward_func()[0].item()

                    scale = reward_cfg["reward_scales"].get(name, 1.0)  # Default to 1.0 if no scale
                    reward_values[name] = raw_value * scale

                visualizer.update(reward_values)
    else:
        obs, _ = env.reset()
        with torch.no_grad():
            while True:
                actions = policy(obs)
                env.commands[0] = torch.tensor([0.5, 0.0, 0.0], device=actions.device)
                obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
