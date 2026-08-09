import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from envs.quadruped_env import QuadrupedFaultEnv


def make_env():
    return QuadrupedFaultEnv(render=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=4, help="Parallel envs; keep small on a laptop CPU")
    parser.add_argument("--save_path", type=str, default="models/base_policy")
    parser.add_argument("--log_dir", type=str, default="logs/base_policy")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy bonus coefficient; higher values encourage more exploration")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed. Set explicitly for every run -- results are not "
                             "comparable across runs without it.")
    args = parser.parse_args()

    set_random_seed(args.seed)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if os.path.exists(args.save_path + ".zip"):
        print(f"WARNING: {args.save_path}.zip already exists and will be OVERWRITTEN "
              f"when this run finishes. Ctrl+C now and pass a different --save_path "
              f"if you meant to keep it.")

    print(f"Training with seed={args.seed}, timesteps={args.timesteps}, "
          f"n_envs={args.n_envs}, ent_coef={args.ent_coef}")
    
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

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
        ent_coef=args.ent_coef,
        seed=args.seed,
        tensorboard_log=args.log_dir,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix=os.path.basename(args.save_path) + "_ckpt",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)
    model.save(args.save_path)
    print(f"Saved base policy to {args.save_path}.zip (seed={args.seed})")


if __name__ == "__main__":
    main()
