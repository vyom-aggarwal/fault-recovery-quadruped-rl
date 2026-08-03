#Watching the model in action

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from envs.quadruped_env import QuadrupedFaultEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/base_policy")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--render", action="store_true", help="Open a PyBullet GUI window")
    parser.add_argument("--fault", type=str, default=None,
                         help="e.g. torque_limit, joint_lock, actuation_delay, sensor_dropout, sensor_noise")
    parser.add_argument("--fault_step", type=int, default=150, help="Step at which to trigger the fault")
    parser.add_argument("--fault_severity", type=float, default=0.2)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = QuadrupedFaultEnv(render=args.render)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        total_reward = 0.0
        fell = False
        fault_triggered = False

        for step in range(env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if args.fault and not fault_triggered and step == args.fault_step:
                print(f"  [step {step}] triggering fault: {args.fault} (severity={args.fault_severity})")
                env.trigger_fault(args.fault, severity=args.fault_severity)
                fault_triggered = True

            if args.render:
                time.sleep(1.0 / 60.0)

            if terminated:
                fell = True
                break
            if truncated:
                break

        status = "FELL" if fell else "stayed standing"
        print(f"Episode {ep}: {step+1} steps, total_reward={total_reward:.2f}, {status}")

    env.close()


if __name__ == "__main__":
    main()
