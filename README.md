# Fault Recovery in RL Locomotion Through Online Residual Adaptation

Research project at the **UC Santa Cruz AIEA Lab**, investigating whether a small residual
correction module that updates *during execution* can help a quadruped recover from unexpected
actuator and sensor faults — without retraining its base locomotion policy.

Everything runs in simulation (PyBullet), on CPU, on a laptop.

---

## Project status

| Component | State |
|---|---|
| Fault-injectable quadruped environment | **Built** — [`envs/quadruped_env.py`](envs/quadruped_env.py) |
| PPO base policy, trained across 5 seeds | **Built** — `models/seed_0..4.zip` (committed) |
| Baseline A (fault, *no* adaptation) evaluation | **Built + run** — see [Results](#results-so-far-baseline-a) |
| Across-seed aggregation & headroom analysis | **Built + run** — [`logs/across_seed_summary.csv`](logs/across_seed_summary.csv) |
| Residual adaptation module | **Not built yet** |
| Baseline B (full policy retraining) | **Not built yet** |
| Held-out fault generalization test (H3) | **Not built yet** |

The headline finding so far is a **ceiling effect**: at the current fault severities, the
unadapted base policy already "recovers" in essentially every trial, which leaves an adaptation
method almost nothing to improve on. Fixing that — harder faults and a stricter recovery
metric — is the immediate next step. Details in [Known limitations](#known-limitations--open-issues).

---

## Contents

- [Abstract](#abstract)
- [Hypotheses](#hypotheses)
- [Results so far (Baseline A)](#results-so-far-baseline-a)
- [Quickstart](#quickstart)
- [Repository layout](#repository-layout)
- [The environment](#the-environment)
- [The fault model](#the-fault-model)
- [Baseline A protocol](#baseline-a-protocol)
- [Full reproduction pipeline](#full-reproduction-pipeline)
- [How the code fits together](#how-the-code-fits-together)
- [Known limitations & open issues](#known-limitations--open-issues)
- [Roadmap](#roadmap)

---

## Abstract

Most reinforcement learning controllers for legged robots are trained assuming the robot is
working perfectly. In the real world that assumption doesn't hold: motors weaken, joints seize,
and sensors start reporting inaccurate values. A policy trained only under ideal conditions
often struggles when that happens, because it has never learned how to respond to failure.

This project asks whether a small residual correction module, updated online, can recover
locomotion performance after an unexpected hardware fault without retraining the original
policy. A quadruped locomotion policy is trained in PyBullet with PPO (Stable-Baselines3), then
actuator and sensor faults are injected mid-rollout. Online residual adaptation is compared
against two baselines: **no adaptation** (Baseline A) and **full policy retraining**
(Baseline B). Staying entirely in simulation makes it possible to test failure modes
systematically and repeatably, including ones too risky or impractical to induce on hardware.

## Hypotheses

- **H1** — Residual adaptation recovers most of the lost locomotion performance within a small,
  bounded number of post-fault timesteps.
- **H2** — It does so without retraining the base policy, landing between Baseline A
  (no adaptation) and Baseline B (full retraining).
- **H3** — Recovery generalizes to fault types the residual module never trained on.

A subset of fault categories (or specific severity levels within a category) is held out
exclusively for H3. The adaptation module never sees them during its own training.

---

## Results so far (Baseline A)

Base policy, **no adaptation**, fault injected mid-episode. Source:
[`logs/across_seed_summary.csv`](logs/across_seed_summary.csv).

**The unit of analysis is the seed.** Each seed contributes one number per metric; the ± is the
spread *across seeds*, not across trials. Four seeds contributed fault-evaluation data (n = 4).

| Fault (severity) | Recovery rate | Fall rate | Recovery time (s) | Post-fault distance (m) |
|---|---|---|---|---|
| `torque_limit` (0.2×) | 100% ± 0% | **36.5% ± 11.7%** | 0.04 ± 0.05 | 1.78 ± 0.25 |
| `joint_lock` | 100% ± 0% | 0% ± 0% | 0.16 ± 0.26 | 2.27 ± 0.26 |
| `actuation_delay` (10 steps) | 100% ± 0% | 0% ± 0% | 0.05 ± 0.09 | 1.64 ± 0.10 |
| `sensor_dropout` | 100% ± 0% | 0% ± 0% | 0.05 ± 0.06 | 2.96 ± 0.19 |
| `sensor_noise` (σ = 0.3) | 100% ± 0% | 0% ± 0% | 0.10 ± 0.14 | 3.04 ± 0.20 |

### Reading these numbers honestly

Three things stand out, and none of them are good news for the experiment as currently designed:

1. **Every fault is at the recovery ceiling.** A 100% baseline recovery rate means there is no
   headroom for *any* adaptation method to demonstrate an improvement. `aggregate_seeds.py`
   flags this automatically (`CEILING -- no method can improve on this; weak choice for H1`).
2. **`torque_limit` scores 100% recovery and 36.5% falls at the same time.** A trial can be
   marked "recovered" and then fall over. That is only possible because recovery is scored as a
   *momentary* velocity crossing, not a sustained return to the pre-fault gait.
3. **Recovery times of 0.04–0.16 s are 2–10 control steps.** At 60 Hz control, the policy is
   being credited with recovering almost before the fault has had time to propagate through the
   dynamics. This strongly suggests the metric fires on transient velocity noise.

The load-bearing conclusion is therefore about the *measurement*, not the policy: the current
recovery criterion is too permissive and the current severities are too mild. Both need to
change before Baseline A can serve as a meaningful comparison point. See
[Known limitations](#known-limitations--open-issues).

The one signal that does survive is the **fall rate**: `torque_limit` at 0.2× nominal is the
only fault that reliably knocks the robot over, making it the strongest current candidate for
the H1 test case.

---

## Quickstart

### 1. Clone

```bash
git clone https://github.com/vyom-aggarwal/fault-recovery-quadruped-rl
```

### 2. Install

From inside the repository:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

On macOS/Linux use `source venv/bin/activate` and `python3` in place of `python`.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install pybullet stable-baselines3 gymnasium
```

Everything here is CPU-only by design — the networks are small (two hidden layers of 128) and
PyBullet is the bottleneck, so a GPU build of torch buys nothing.

> `pybullet` compiles a C++ physics core during install. The first `pip install` can take
> several minutes with no output. That is normal, not a hang.

The robot model (`laikago.urdf`) and ground plane ship inside `pybullet_data` — there are no
assets to download separately.

### 3. Verify the install

```bash
python scripts/smoke_test.py
```

Expected: observation/action shapes, a joint count of 12, and confirmation that a mid-episode
fault injection doesn't crash the sim.

### 4. Reproduce the Baseline A results without training anything

The five trained policies are committed (~550 KB each), so the evaluation half of the pipeline
runs from a clean clone in minutes:

```bash
python scripts/run_multiseed.py --skip_training --trials 100
```

```bash
python scripts/aggregate_seeds.py
```

This regenerates `results/seed_*/baseline_fault_results.csv`, `results/manifest.json`, and
`logs/across_seed_summary.csv`.

---

## Repository layout

```
envs/
  quadruped_env.py          the ONLY file with real robot logic:
                            physics, observations, reward, fault injection

scripts/
  smoke_test.py             does the env run at all?                    (seconds)
  check_reset_pose.py       is the starting stance sane?  [GUI]         (seconds)
  train_base_policy.py      PPO training for one seed                   (~45-50 min)
  evaluate_policy.py        roll out a policy, optionally with GUI      (seconds)
  diagnose_gait.py          is it REALLY walking, numerically?          (seconds)
  baseline_fault_eval.py    Baseline A: inject faults, log trials       (minutes)
  analyze_baseline.py       summarize ONE seed's trial CSV
  aggregate_seeds.py        summarize ACROSS seeds + headroom analysis
  run_multiseed.py          orchestrates the whole per-seed pipeline

models/
  seed_0.zip .. seed_4.zip  trained base policies (committed)

logs/
  across_seed_summary.csv   committed across-seed results
  seed_*/progress.csv       per-seed training curves (generated, not committed)

results/                    generated, not committed
  manifest.json             per-seed status + gait-check metrics
  seed_*/baseline_fault_results.csv   trial-level Baseline A data
```

**What is and isn't committed:** the trained policies and the final across-seed summary are in
the repo. The trial-level CSVs, training curves, and `manifest.json` are regenerated by the
commands in the [Quickstart](#quickstart). There is no `.gitignore` yet, so generated files show
up in `git status` — don't commit them by accident.

---

## The environment

`QuadrupedFaultEnv` is a standard Gymnasium environment. One instance owns one PyBullet client.

| Property | Value |
|---|---|
| Robot | `laikago.urdf` from `pybullet_data`, 12 revolute joints |
| Physics rate | 240 Hz (`setTimeStep = 1/240`) |
| Action repeat | 4 → **control rate 60 Hz** |
| Episode length | 1000 control steps ≈ 16.7 s of sim time |
| Observation | 34-dim `Box(-inf, inf)` |
| Action | 12-dim `Box(-1, 1)` |
| Target velocity | 0.5 m/s forward, flat terrain |

### Observation (34-dim)

| Slice | Contents |
|---|---|
| `0:12` | joint angles (rad) — **the only channel faults corrupt** |
| `12:24` | joint velocities (rad/s) |
| `24:28` | base orientation quaternion |
| `28:31` | base linear velocity |
| `31:34` | base angular velocity |

### Action

Actions are **offsets around a fixed standing pose**, not raw torques:

```
target_angle[i] = standing_pose[i] + action[i] * 0.5        # standing_pose = [0.0, -0.7, 0.7] * 4
```

Each joint is then driven by PyBullet `POSITION_CONTROL` with a force ceiling of 20 N·m. This
matters for interpreting the `torque_limit` fault: it caps the position controller's force
budget, which is not identical to saturating a true torque command.

### Reward

```
r = -|v_forward - 0.5|                  velocity tracking
    - 0.03 * Σ (τ_i / 20)²              energy penalty
    - (1 - up_z)²                       stay upright
    - max(0, 0.50 - height)             don't crouch
    + 1.0                               alive bonus
    - 10.0                              one-time penalty on falling
```

`v_forward` is the base linear velocity projected onto the robot's forward axis, and `up_z` is
the world-frame z-component of its up axis — both derived from the rotation matrix in
`_get_orientation_frame`, because Laikago's URDF frame is not axis-aligned in the obvious way.

### Reset and termination

- **Reset** snaps every joint directly to the standing pose (rather than starting at zero and
  letting the policy fight its way up), then runs 60 settle steps under position control before
  handing back the first observation. Start height 0.48 m.
- **Terminated** when base height < 0.35 m **or** `up_z` < 0.5 (roughly 60° from vertical).
- **Reset clears all faults.** Faults must be triggered *after* `reset()`, mid-episode.

---

## The fault model

Call `env.trigger_fault(...)` at any point during an episode.

| `fault_type` | Scope | `severity` means | Mechanism |
|---|---|---|---|
| `torque_limit` | one joint | fraction of nominal (e.g. `0.2`) | scales that joint's 20 N·m force ceiling |
| `joint_lock` | one joint | *ignored* | freezes the joint at its angle at injection time |
| `actuation_delay` | all joints | integer steps of lag (e.g. `10`) | actions queue through a FIFO buffer |
| `sensor_dropout` | one joint | *ignored* | that joint's **angle** reading is forced to 0 |
| `sensor_noise` | all joints | Gaussian σ (e.g. `0.3`) | noise added to all **angle** readings |

```python
env.trigger_fault("torque_limit", joint=3, severity=0.2)   # specific joint
env.trigger_fault("torque_limit", severity=0.2)            # random joint, drawn from env.np_random
env.trigger_fault("sensor_noise", severity=0.3)            # global fault, joint ignored
env.clear_faults()                                          # revert to healthy
```

`joint=None` on a per-joint fault draws a uniform random joint from `env.np_random`, so the
choice is reproducible given the episode seed.

Two properties are worth keeping in mind when designing experiments:

- Sensor faults corrupt **joint angles only**. Joint velocities, the base quaternion, and the
  base velocities pass through clean, so the policy always retains an uncorrupted view of its
  own body pose. That is a generous assumption and a reasonable knob to tighten later.
- Faults **compose**. Calling `trigger_fault` twice with different types leaves both active;
  `active_fault` only records the most recent one.

### Extension points

Two hooks exist in `envs/quadruped_env.py` but are currently dormant, and both are the natural
attachment points for the residual work:

- `_apply_fault_to_action(action)` — an identity pass-through today. This is where an
  action-space fault (or the residual correction itself) would be applied.
- `_sensor_bias` — allocated and added to joint angles every step, but no fault type currently
  sets it. Wiring up a `sensor_bias` / drift fault is a few lines.

---

## Baseline A protocol

Implemented in [`scripts/baseline_fault_eval.py`](scripts/baseline_fault_eval.py). One trial:

| Phase | Steps | What happens |
|---|---|---|
| Pre-fault | 0 – 199 (3.3 s) | policy walks; trial is **discarded** if it falls here |
| Baseline measurement | last 50 pre-fault steps | `baseline_vel` = mean forward velocity |
| Injection | step 200 | `trigger_fault(type, severity)`, random joint |
| Post-fault window | 300 steps (5 s) | recovery and fall are scored |

**Recovery** is the first post-fault step where

```
|v_forward - baseline_vel| ≤ 0.15 * |baseline_vel|
```

recorded in seconds (`step / 60`). A trial that never satisfies this leaves `recovery_time_s`
empty, which downstream scripts count as *not recovered*.

Default severities under test:

```python
torque_limit 0.2 · joint_lock 1.0 · actuation_delay 10 · sensor_dropout 1.0 · sensor_noise 0.3
```

Trial `i` uses seed `i`, and the same seed set is reused across every fault type, so fault types
are compared on matched initial conditions.

---

## Full reproduction pipeline

### Cheap sanity checks first

These take seconds and catch mistakes that otherwise cost an hour of training. Don't skip them.

```bash
python scripts/smoke_test.py
```

```bash
python scripts/check_reset_pose.py
```

`check_reset_pose.py` opens a PyBullet GUI window and waits on Enter before closing.

### Train and evaluate all seeds

```bash
python scripts/run_multiseed.py --seeds 5 --timesteps 500000 --n_envs 4 --trials 100
```

For each seed this runs training → a gait check → fault evaluation, writing
`results/manifest.json` incrementally so a crash mid-run doesn't lose completed seeds. Existing
`models/seed_N.zip` files cause training to be skipped for that seed — delete a model to force a
retrain, or pass `--skip_training` to evaluate only.

**Budget roughly 45–50 minutes per seed** at 500k timesteps with `--n_envs 4` on a laptop CPU,
so about 4 hours for all five.

The **gait check** between training and evaluation is the quality gate. A seed counts as
converged only if it both survives ≥ 90% of a 1000-step rollout *and* averages ≥ 0.2 m/s. Both
conditions are needed: a policy can stand still forever (survives, no displacement) or lunge and
fall (displacement, no survival). Seeds that fail are excluded from fault evaluation and counted
against the reported convergence rate.

### Aggregate

```bash
python scripts/aggregate_seeds.py
```

Prints training reliability (how many seeds converged), per-fault across-seed statistics, a
high-variance warning when the per-seed recovery spread exceeds 40 points, and a headroom
ranking that classifies each fault as `CEILING` / `moderate headroom` / `large headroom`. Writes
[`logs/across_seed_summary.csv`](logs/across_seed_summary.csv).

### Inspect a single policy

```bash
python scripts/evaluate_policy.py --model models/seed_0 --render --episodes 3
```

```bash
python scripts/evaluate_policy.py --model models/seed_0 --fault torque_limit --fault_step 150 --fault_severity 0.2
```

```bash
python scripts/diagnose_gait.py --model models/seed_0 --steps 300
```

`diagnose_gait.py` is the numerical counterpart to watching the GUI: per-joint angle ranges
(a real gait needs tens of degrees of swing, not a few), ground-contact variation (a constant
count of 4 means all feet stay planted and the robot is sliding), and net displacement versus
target speed.

> **Gotcha:** several scripts still default to `--model models/base_policy`, which no longer
> exists — the multi-seed refactor replaced it with `models/seed_N`. Always pass `--model`
> explicitly. Likewise `analyze_baseline.py` defaults to `--csv logs/baseline_fault_results.csv`;
> point it at `results/seed_N/baseline_fault_results.csv` instead.

### Training options worth knowing

```bash
python scripts/train_base_policy.py --seed 0 --timesteps 500000 --save_path models/seed_0 --log_dir logs/seed_0
```

| Flag | Default | Notes |
|---|---|---|
| `--seed` | `0` | Set it explicitly on every run. Results are not comparable without it. |
| `--timesteps` | `500_000` | |
| `--n_envs` | `4` | Parallel envs; keep small on a laptop CPU. |
| `--ent_coef` | `0.01` | Entropy bonus; raise to encourage exploration. |
| `--log_format` | `csv` | `csv` writes a directly plottable `progress.csv`. `tensorboard` uses an async writer thread that has proven fragile on Windows — it can die mid-run and take training down with it. |

PPO hyperparameters: `n_steps=2048`, `batch_size=256`, `n_epochs=10`, `lr=3e-4`, `γ=0.99`,
`gae_lambda=0.95`, `clip_range=0.2`, `net_arch=[128, 128]`. Checkpoints are written every
~50k timesteps alongside the final model.

---

## How the code fits together

Exactly one file contains real quadruped logic. Everything in `scripts/` is a thin driver that
imports it and does one job. Dependencies point one way only: scripts import the environment;
the environment imports nothing from scripts.

```
                          envs/quadruped_env.py
             physics · observations · reward · fault injection
                                    │
      ┌──────────┬───────────┬──────┴─────┬─────────────┬──────────────┐
      │          │           │            │             │              │
 smoke_test  check_reset  train_base  evaluate_    diagnose_    baseline_fault
             _pose        _policy     policy       gait         _eval
      │          │           │            │             │              │
 "does it    "is the      "learn to   "watch it"   "is it REALLY   "how does it
  run?"       pose sane?"  walk"                    walking?"       fail?"


 run_multiseed.py    ── orchestrates ──▶  train → gait check → fault eval, per seed
 analyze_baseline.py ── reduces ──▶       one seed's trials  → console + summary CSV
 aggregate_seeds.py  ── reduces ──▶       all seeds' trials  → logs/across_seed_summary.csv
```

---

## Known limitations & open issues

These are stated plainly because they determine what the next round of experiments has to fix.

1. **Ceiling effect at current severities.** Baseline A recovers 100% of the time on all five
   fault types, leaving no headroom to demonstrate that residual adaptation helps. Severities
   need to increase (or faults need to compound) until baseline recovery lands somewhere in the
   30–70% range.
2. **The recovery metric measures a momentary crossing, not sustained recovery.** Recovery times
   of 2–10 control steps, and `torque_limit` scoring 100% recovery alongside a 36.5% fall rate,
   both point the same way. A defensible criterion should require the velocity band to be held
   for a sustained window, and should not credit trials that subsequently fall.
3. **The recovery band is relative to `baseline_vel`.** A trial whose pre-fault speed happens to
   be near zero gets a near-zero tolerance band; a fast trial gets a wide one. Trials are not
   being held to a comparable standard.
4. **n = 4 seeds.** Five policies are trained and committed, but only four contributed fault
   data to the current summary. Four seeds is thin for any variance claim.
5. **Sensor faults touch joint angles only.** Velocities, base orientation, and base velocities
   are never corrupted, so the policy always keeps a clean view of its body state.
6. **`torque_limit` caps a position controller's force budget**, which is related to but not the
   same as true torque saturation on a torque-controlled actuator.
7. **Single robot, flat terrain, simulation only.** No terrain variation, no domain
   randomization, no sim-to-real claim is being made.
8. **`check_reset_pose.py` prints "expected ~0.30" for the reset height**, while the environment
   treats anything below 0.35 m as fallen. One of the two numbers is stale and they should be
   reconciled.

---

## Roadmap

- [ ] Redefine the recovery criterion (sustained window; falls disqualify recovery).
- [ ] Sweep fault severities to find operating points with real headroom.
- [ ] Implement the residual adaptation module on the `_apply_fault_to_action` hook.
- [ ] Implement Baseline B (full policy retraining post-fault) for the upper-bound comparison.
- [ ] Choose the held-out fault set for H3 — it needs both genuine headroom *and* structural
      difference from the training faults (physics-side vs. sensor-side).
- [ ] Add a `sensor_bias` / drift fault using the existing `_sensor_bias` state.
- [ ] Expand to more seeds once the metric is fixed and re-run the whole pipeline.
- [ ] Pin dependency versions in a `requirements.txt`; add a `.gitignore` for `results/` and
      `logs/seed_*/`.

---

## Notes

No license file is currently specified — please get in touch before reusing this work.
Conducted through the UC Santa Cruz AIEA Lab.
