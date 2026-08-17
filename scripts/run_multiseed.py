import argparse
import json
import os
import subprocess
import sys
import time
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import numpy as np
 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 
 
def gait_check(model_path, steps=1000, target_velocity=0.5, tol=0.30):
    import pybullet as p
    from stable_baselines3 import PPO
    from envs.quadruped_env import QuadrupedFaultEnv
 
    model = PPO.load(model_path)
    env = QuadrupedFaultEnv(render=False)
    obs, info = env.reset(seed=0)
 
    start_pos = np.array(p.getBasePositionAndOrientation(
        env.robot_id, physicsClientId=env._client)[0])
    joint_history = []
    contact_counts = []
    fell = False
 
    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        joint_history.append([
            p.getJointState(env.robot_id, j, physicsClientId=env._client)[0]
            for j in env.joint_ids
        ])
        contact_counts.append(len(p.getContactPoints(
            env.robot_id, env.plane_id, physicsClientId=env._client)))
        if terminated:
            fell = True
            break
        if truncated:
            break
 
    end_pos = np.array(p.getBasePositionAndOrientation(
        env.robot_id, physicsClientId=env._client)[0])
    env.close()
 
    joint_history = np.array(joint_history)
    steps_survived = step + 1
    elapsed = steps_survived * 4 / 240.0          # action_repeat=4, 240 Hz physics
    displacement = float(np.linalg.norm(end_pos - start_pos))
    speed = displacement / elapsed if elapsed > 0 else 0.0
    joint_range = float((joint_history.max(axis=0) - joint_history.min(axis=0)).mean()) \
        if len(joint_history) else 0.0
    contact_var = float(np.mean(np.diff(contact_counts) != 0)) if len(contact_counts) > 1 else 0.0

    survived = steps_survived >= 0.9 * steps
    tracking_error = abs(speed - target_velocity) / target_velocity if target_velocity else 1.0
    converged = survived and (tracking_error <= tol)
 
    return {
        "converged": bool(converged),
        "survived_episode": bool(survived),
        "tracking_error": round(tracking_error, 4),
        "steps_survived": int(steps_survived),
        "fell": bool(fell),
        "mean_speed_mps": round(speed, 4),
        "displacement_m": round(displacement, 4),
        "mean_joint_range_rad": round(joint_range, 4),
        "contact_variation": round(contact_var, 4),
    }
 
 
def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode == 0
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds (0..N-1)")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--trials", type=int, default=100, help="Fault trials per fault type, per seed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--skip_training", action="store_true",
                        help="Evaluate existing policies without retraining")
    parser.add_argument("--gait_steps", type=int, default=1000)
    parser.add_argument("--target_velocity", type=float, default=0.5,
                        help="Commanded velocity the base policy was trained to track.")
    parser.add_argument("--convergence_tol", type=float, default=0.30,
                        help="A seed counts as converged only if it survives the episode AND "
                             "tracks the commanded velocity within this fraction. The old "
                             "criterion (speed >= 0.2 absolute) admitted policies running at "
                             "60%% of target, whose fault responses are not comparable to "
                             "on-target policies.")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Retrain even if a model already exists at the target path.")
    parser.add_argument("--log_format", type=str, default="csv",
                        choices=["csv", "tensorboard", "none"],
                        help="Passed through to training. csv is the robust default.")
    args = parser.parse_args()
 
    os.makedirs(args.results_dir, exist_ok=True)
    manifest_path = os.path.join(args.results_dir, "manifest.json")
    manifest = {"config": vars(args), "seeds": {}}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            manifest["config"] = vars(args)
        except json.JSONDecodeError:
            pass
 
    started = time.time()
 
    for seed in range(args.seeds):
        print("\n" + "=" * 70)
        print(f"SEED {seed} / {args.seeds - 1}")
        print("=" * 70)
 
        model_path = os.path.join(args.models_dir, f"seed_{seed}")
        seed_results_dir = os.path.join(args.results_dir, f"seed_{seed}")
        os.makedirs(seed_results_dir, exist_ok=True)
 
        # train 
        if not args.skip_training:
            existing_cfg_path = model_path + "_trainconfig.json"
            stale = False
            if os.path.exists(model_path + ".zip") and not args.force_retrain:
                if os.path.exists(existing_cfg_path):
                    with open(existing_cfg_path) as f:
                        prev = json.load(f)
                    mismatches = {k: (prev.get(k), v) for k, v in
                                  (("timesteps", args.timesteps),
                                   ("ent_coef", args.ent_coef),
                                   ("n_envs", args.n_envs))
                                  if prev.get(k) != v}
                    if mismatches:
                        stale = True
                        print(f"[seed {seed}] EXISTING MODEL WAS TRAINED WITH DIFFERENT SETTINGS:")
                        for k, (was, now) in mismatches.items():
                            print(f"    {k}: existing={was}  requested={now}")
                        print(f"  Mixing training budgets across seeds confounds the experiment.")
                        print(f"  Retraining this seed with the requested settings.")
                else:
                    stale = True
                    print(f"[seed {seed}] existing model has no _trainconfig.json -- provenance "
                          f"unknown, so it cannot be trusted to match this batch. Retraining.")
 
            if os.path.exists(model_path + ".zip") and not stale and not args.force_retrain:
                print(f"[seed {seed}] {model_path}.zip already exists with matching settings "
                      f"-- skipping training.")
            else:
                ok = run([sys.executable, "scripts/train_base_policy.py",
                          "--seed", str(seed),
                          "--timesteps", str(args.timesteps),
                          "--n_envs", str(args.n_envs),
                          "--ent_coef", str(args.ent_coef),
                          "--save_path", model_path,
                          "--log_dir", os.path.join("logs", f"seed_{seed}"),
                          "--log_format", args.log_format])
                if not ok:
                    print(f"[seed {seed}] TRAINING FAILED -- skipping this seed")
                    manifest["seeds"][str(seed)] = {"status": "train_failed"}
                    continue
 
        if not os.path.exists(model_path + ".zip"):
            print(f"[seed {seed}] no model at {model_path}.zip -- skipping")
            manifest["seeds"][str(seed)] = {"status": "missing_model"}
            continue
 
        # gait check
        print(f"\n[seed {seed}] checking whether the policy walks...")
        try:
            gait = gait_check(model_path, steps=args.gait_steps,
                              target_velocity=args.target_velocity,
                              tol=args.convergence_tol)
        except Exception as exc:
            print(f"[seed {seed}] gait check errored: {exc}")
            manifest["seeds"][str(seed)] = {"status": "gait_check_failed", "error": str(exc)}
            continue
 
        status = "converged" if gait["converged"] else "NOT converged"
        print(f"[seed {seed}] {status}: {gait['mean_speed_mps']} m/s "
              f"(tracking error {gait['tracking_error']:.0%} vs {args.target_velocity} m/s target), "
              f"{gait['steps_survived']}/{args.gait_steps} steps, "
              f"joint range {gait['mean_joint_range_rad']} rad")
 
        manifest["seeds"][str(seed)] = {"status": "ok", "gait": gait,
                                        "model_path": model_path + ".zip"}
 
        # A non-converged seed is a REPORTABLE RESULT (convergence rate), not a failure to hide, but we still skip the eval
        if not gait["converged"]:
            print(f"[seed {seed}] skipping fault evaluation -- policy did not learn to walk. "
                  f"This counts toward the reported convergence rate.")
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            continue
 
        # ---- fault evaluation ------------------------------------------
        out_csv = os.path.join(seed_results_dir, "baseline_fault_results.csv")
        ok = run([sys.executable, "scripts/baseline_fault_eval.py",
                  "--model", model_path,
                  "--trials", str(args.trials),
                  "--out", out_csv])
        manifest["seeds"][str(seed)]["baseline_csv"] = out_csv if ok else None
 
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
 
    # ---- summary -------------------------------------------------------
    elapsed_min = (time.time() - started) / 60.0
    ok_seeds = [s for s, v in manifest["seeds"].items() if v.get("status") == "ok"]
    converged = [s for s in ok_seeds if manifest["seeds"][s]["gait"]["converged"]]
 
    print("\n" + "=" * 70)
    print(f"DONE in {elapsed_min:.1f} min")
    print(f"Seeds run: {len(ok_seeds)} | converged to a walking gait: "
          f"{len(converged)}/{len(ok_seeds)}")
    if ok_seeds:
        speeds = [manifest["seeds"][s]["gait"]["mean_speed_mps"] for s in converged]
        if speeds:
            print(f"Converged-seed speed: {np.mean(speeds):.3f} +/- "
                  f"{np.std(speeds, ddof=1) if len(speeds) > 1 else 0.0:.3f} m/s")
    print(f"Manifest: {manifest_path}")
    print("\nNext: python scripts/aggregate_seeds.py")
 
 
if __name__ == "__main__":
    main()