import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
from stable_baselines3 import PPO

from envs.quadruped_env import QuadrupedFaultEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/seed_0")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    model = PPO.load(args.model)
    env = QuadrupedFaultEnv(render=False)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        start = np.array(p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env._client)[0])

        fwd_vels, lat_vels, yaws = [], [], []

        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            _, orn = p.getBasePositionAndOrientation(
                env.robot_id, physicsClientId=env._client)
            lin_vel, _ = p.getBaseVelocity(env.robot_id, physicsClientId=env._client)
            rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

            # local Y = up, local Z = forward, local X = lateral.
            forward_axis = rot[:, 2]
            lateral_axis = rot[:, 0]

            fwd_vels.append(float(np.dot(lin_vel, forward_axis)))
            lat_vels.append(float(np.dot(lin_vel, lateral_axis)))
            # heading angle in the ground plane, for cumulative turn
            yaws.append(np.arctan2(forward_axis[1], forward_axis[0]))

            if terminated or truncated:
                break

        end = np.array(p.getBasePositionAndOrientation(
            env.robot_id, physicsClientId=env._client)[0])
        n = step + 1
        elapsed = n * env.action_repeat / 240.0

        net_disp = float(np.linalg.norm(end - start))
        net_speed = net_disp / elapsed
        mean_fwd = float(np.mean(fwd_vels))
        mean_abs_lat = float(np.mean(np.abs(lat_vels)))
        heading_change = float(np.degrees(np.unwrap(yaws)[-1] - np.unwrap(yaws)[0]))

        print(f"\n--- episode {ep} ({n} steps, {elapsed:.1f}s) ---")
        print(f"  net displacement speed : {net_speed:.3f} m/s   <- what gait_check reports")
        print(f"  mean FORWARD velocity  : {mean_fwd:.3f} m/s   <- what the reward tracks")
        print(f"  mean |lateral| velocity: {mean_abs_lat:.3f} m/s")
        print(f"  net heading change     : {heading_change:+.1f} deg")

        ratio = mean_abs_lat / abs(mean_fwd) if abs(mean_fwd) > 1e-6 else float("inf")
        gap = abs(net_speed - mean_fwd)
        if ratio > 0.15:
            print(f"  -> DRIFTING: lateral motion is {ratio:.0%} of forward motion.")
            print(f"     Reported speed and tracked velocity are not the same quantity.")
        elif gap > 0.05:
            print(f"  -> {gap:.3f} m/s gap between the two measures; worth explaining.")
        else:
            print(f"  -> walks essentially straight; the two measures agree.")

    env.close()
    print("\nIf drift is significant, the fix is a lateral-velocity penalty in the")
    print("reward -- but that changes the reward function, so only do it if you are")
    print("prepared to retrain. Otherwise, report FORWARD velocity (not net")
    print("displacement) as the tracked quantity and note the distinction.")


if __name__ == "__main__":
    main()
