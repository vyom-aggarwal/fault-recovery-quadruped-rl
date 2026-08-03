"""
Phase 1 training script: base locomotion policy, no faults.

Run:
    python scripts/train_base_policy.py --timesteps 500000

This trains PPO (Schulman et al., 2017) via Stable-Baselines3 (Raffin et al., 2021)
on the QuadrupedFaultEnv (PyBullet / Coumans & Bai) with no faults active.
The resulting policy is the "frozen base policy" that later scripts will
subject to fault injection and residual adaptation.
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.quadruped_env import QuadrupedFaultEnv


def make_env():
    return QuadrupedFaultEnv(render=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=4, help="Parallel envs; keep small on a laptop CPU")
    parser.add_argument("--save_path", type=str, default="models/base_policy")
    parser.add_argument("--log_dir", type=str, default="logs/base_policy")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    vec_env = make_vec_env(make_env, n_envs=args.n_envs)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(net_arch=[128, 128]),  # small net — keeps CPU training fast
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix="base_policy_ckpt",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)
    model.save(args.save_path)
    print(f"Saved base policy to {args.save_path}.zip")


if __name__ == "__main__":
    main()
