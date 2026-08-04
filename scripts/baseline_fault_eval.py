import argparse
import csv
import os
import sys
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
from stable_baselines3 import PPO
from envs.quadruped_env import QuadrupedFaultEnv
 
FAULT_CONFIGS = [
    {"type": "torque_limit", "severity": 0.2},
    {"type": "joint_lock", "severity": 1.0},
    {"type": "actuation_delay", "severity": 10},
    {"type": "sensor_dropout", "severity": 1.0},
    {"type": "sensor_noise", "severity": 0.3},
]
 
FAULT_ONSET_STEP = 200          
POST_FAULT_WINDOW = 300         
RECOVERY_VEL_TOLERANCE = 0.15   
 
 
def run_trial(model, env, fault_config, seed):
    obs, info = env.reset(seed=seed)
 
    pre_fault_vels = []
    for step in range(FAULT_ONSET_STEP):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        pre_fault_vels.append(info["forward_vel"])
        if terminated or truncated:
            return None  # fell before the fault was even injected; discard this trial
    baseline_vel = np.mean(pre_fault_vels[-50:])  # settled walking speed just before the fault
 
    env.trigger_fault(fault_config["type"], severity=fault_config["severity"])
 
    post_fault_start_pos = None
    recovery_time = None  # None = never recovered within the window
    fell = False
    distance_start = None
 
    for step in range(POST_FAULT_WINDOW):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
 
        if distance_start is None:
            import pybullet as p
            distance_start = np.array(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)[0])
 
        if recovery_time is None and abs(info["forward_vel"] - baseline_vel) <= RECOVERY_VEL_TOLERANCE * abs(baseline_vel):
            recovery_time = step / 60.0
 
        if terminated:
            fell = True
            break
 
    import pybullet as p
    end_pos = np.array(p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)[0])
    post_fault_distance = np.linalg.norm(end_pos - distance_start) if distance_start is not None else 0.0
 
    return {
        "fault_type": fault_config["type"],
        "severity": fault_config["severity"],
        "seed": seed,
        "baseline_vel": baseline_vel,
        "recovery_time_s": recovery_time if recovery_time is not None else "",
        "post_fault_distance_m": post_fault_distance,
        "post_fault_steps_survived": step + 1,
        "fell": fell,
    }
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/base_policy")
    parser.add_argument("--trials", type=int, default=5, help="Trials per fault type (proposal targets 30)")
    parser.add_argument("--out", type=str, default="logs/baseline_fault_results.csv")
    args = parser.parse_args()
 
    model = PPO.load(args.model)
    env = QuadrupedFaultEnv(render=False)
 
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
 
    for fault_config in FAULT_CONFIGS:
        print(f"\n=== {fault_config['type']} (severity={fault_config['severity']}) ===")
        for trial in range(args.trials):
            seed = trial  # same seeds reused across fault types for comparability
            result = run_trial(model, env, fault_config, seed)
            if result is None:
                print(f"  trial {trial}: fell BEFORE fault injection, discarded")
                continue
            results.append(result)
            rt = f"{result['recovery_time_s']:.2f}s" if result["recovery_time_s"] != "" else "never"
            print(f"  trial {trial}: recovery={rt}, "
                  f"post-fault dist={result['post_fault_distance_m']:.2f}m, "
                  f"fell={result['fell']}")
 
    env.close()
 
    if results:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} trial results to {args.out}")
    else:
        print("\nNo valid trials completed -- check that the base policy survives at least "
              f"{FAULT_ONSET_STEP} steps before any fault is injected.")
 
 
if __name__ == "__main__":
    main()