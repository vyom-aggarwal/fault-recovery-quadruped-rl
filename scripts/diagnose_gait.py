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
    parser.add_argument("--model", type=str, default="models/base_policy")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
 
    model = PPO.load(args.model)
    env = QuadrupedFaultEnv(render=False)
    obs, info = env.reset(seed=0)
 
    joint_angle_history = []
    contact_count_history = []
    start_pos = np.array(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)[0])
 
    for step in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
 
        angles = [p.getJointState(env.robot_id, j, physicsClientId=env._client)[0] for j in env.joint_ids]
        joint_angle_history.append(angles)
 
        contacts = p.getContactPoints(env.robot_id, env.plane_id, physicsClientId=env._client)
        contact_count_history.append(len(contacts))
 
        if terminated or truncated:
            break
 
    joint_angle_history = np.array(joint_angle_history)
    contact_count_history = np.array(contact_count_history)
 
    print(f"Ran {step+1} steps before {'falling' if terminated else 'truncating'}.")
    print()
    print("Per-joint angle range over the rollout (max - min, radians):")
    ranges = joint_angle_history.max(axis=0) - joint_angle_history.min(axis=0)
    for idx, r in enumerate(ranges):
        print(f"  joint {idx}: range={r:.4f} rad ({np.degrees(r):.1f} deg)")
    print(f"  --> mean range across all joints: {ranges.mean():.4f} rad ({np.degrees(ranges.mean()):.1f} deg)")
    print("  (a real walking gait typically needs tens of degrees of swing per joint per step cycle;")
    print("   a few degrees or less means the legs are essentially stiff)")
    print()
    print(f"Ground contact points per step: mean={contact_count_history.mean():.2f}, "
          f"min={contact_count_history.min()}, max={contact_count_history.max()}")
    print("  (a proper gait usually shows contact count VARYING as feet lift and land;")
    print("   a constant count near 4 every single step means all feet stay planted the whole time -- sliding)")
    print(f"  fraction of steps with contact count unchanged from previous step: "
          f"{np.mean(np.diff(contact_count_history) == 0):.2%}")
 
    end_pos = np.array(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)[0])
    net_displacement = np.linalg.norm(end_pos - start_pos)
    elapsed_sim_time = (step + 1) * env.action_repeat / 240.0
 
    print()
    print(f"Net displacement over rollout: {net_displacement:.3f} m over {elapsed_sim_time:.2f} s of sim time")
    print(f"  --> average speed: {net_displacement/elapsed_sim_time:.3f} m/s (target was {env.target_velocity} m/s)")
    print("  (real leg motion + no net displacement would mean it's cycling its legs in place,")
    print("   not actually translating forward -- worth checking if this number is small)")
 
    env.close()
 
 
if __name__ == "__main__":
    main()