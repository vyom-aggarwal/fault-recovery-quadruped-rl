"""
Sanity check — run this FIRST, before training anything.

Confirms the environment loads, steps, and resets correctly, and that
fault injection doesn't crash the simulation. Takes a few seconds.

Run:
    python scripts/smoke_test.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.quadruped_env import QuadrupedFaultEnv


def main():
    env = QuadrupedFaultEnv(render=False)
    obs, info = env.reset(seed=0)
    print(f"Observation shape: {obs.shape} (expected {env.observation_space.shape})")
    print(f"Action shape: {env.action_space.shape}")
    print(f"Number of controllable joints found on robot: {env.num_joints}")

    total_reward = 0.0
    for i in range(100):
        action = env.action_space.sample() * 0.1  # small random actions
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if i == 50:
            print("Triggering a torque_limit fault on joint 0 at step 50...")
            env.trigger_fault("torque_limit", joint=0, severity=0.2)
        if terminated or truncated:
            print(f"Episode ended early at step {i} (terminated={terminated})")
            break

    print(f"Ran {i+1} steps, total reward: {total_reward:.3f}")
    print(f"Final height: {info['height']:.3f} (should be > 0.25 if still standing)")
    env.close()
    print("Smoke test passed: environment resets, steps, and accepts fault injection without crashing.")


if __name__ == "__main__":
    main()
