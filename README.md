# Fault Recovery in Reinforcement Learning Locomotion Through Online Residual Adaptation
Research project conducted through the UC Santa Cruz AIEA Lab investigating online residual correction to help quadruped robots recover from unexpected actuator and sensor faults in PyBullet without full policy retraining.

## Abstract
Most reinforcement learning controllers for legged robots are trained assuming the robot is working perfectly. In real-world settings, that assumption doesn't always hold. Motors can weaken, joints can become stuck, and sensors can start reporting inaccurate values. When that happens, a policy trained only under ideal conditions often struggles because it has never learned how to respond to those failures. In this project, I want to investigate whether a small residual correction module that updates during execution can help a robot recover from unexpected hardware failures without retraining its original locomotion policy. I train a quadruped locomotion policy in simulation (PyBullet) using proximal policy optimization (PPO)(1), then inject a range of actuator and sensor faults mid-rollout. I compare an online residual correction against two baselines: no adaptation, and full policy retraining. My central hypothesis is that residual adaptation recovers most of the lost locomotion performance within a small, bounded number of post-fault timesteps, and that this recovery generalizes to fault types the module never trained on. The project stays entirely in simulation, which lets me test failure modes systematically and repeatably — including ones too risky or impractical to induce on real hardware.

## Methodology
### Simulation Environment
Simulator: PyBullet(6), using a standard quadruped model with 12 actuated degrees of freedom.  
 - **Base task:** forward locomotion at a target velocity on flat terrain. The reward combines forward progress, an energy penalty, and stability terms (orientation and height).  
Base policy: trained with PPO(1) via Stable-Baselines3(7), on CPU. I keep episode length and network size small enough to train on a laptop.

### Fault Taxonomy and Injection
I inject faults mid-run, at randomized onset times, drawn from at least these categories:  
 - **Actuator torque limiting** - one joint capped at 20% of its nominal maximum torque
 - **Joint lock** - one joint frozen at its current position
 - **Actuation delay** - commanded torque applied with a fixed lag
 - **Sensor dropout** - a proprioceptive channel intermittently returns stale or zeroed values
 - **Sensor bias/noise** - a joint-angle or IMU reading corrupted with drift or added noise
  
A subset of these categories, or specific severity levels within a category, is held out exclusively for the generalization test in Hypothesis 3. The adaptation module never sees them during its own training.  

## How to Use This
### Clone the Repo
Clone this repository by running this command in your terminal
```bash
git clone https://github.com/vyom-aggarwal/fault-recovery-quadruped-rl
``` 
You should then see the repository in whichever directory your terminal was in. 
### Setup
Just ```cd``` into your repository and run this command.
```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install pybullet stable-baselines3 gymnasium
```

`pybullet` compiles a C++ physics core on install, meaning that the first `pip install`
can take several minutes, which is completely normal.  

### Quick Sanity Check
Run this script to test whether or not your computer has downloaded all the necessary prerequisite libraries to run this project.

```bash
python scripts/smoke_test.py
```

Expected output: observation/action shapes, joint count, and a confirmation
that a mid-episode fault injection doesn't crash the sim.  

## How the Scripts Work

### Mental Model
Exactly one file contains the real quadruped logic: `envs/quadruped_env.py`. Everything in
`scripts/` is a thin driver that imports it and does one job.

```
                      envs/quadruped_env.py
                 (physics + reward + fault injection)
                                |
     +-------------+------------+------------+--------------+
     |             |            |            |              |
smoke_test   check_reset   train_base   evaluate_    diagnose_gait
             _pose         _policy      policy       baseline_fault_eval
     |             |            |            |              |
"does it     "is the      "learn to    "watch it    "is it REALLY
 run?"        pose sane?"  walk"        walk"        walking / how
                                                     does it fail?"
```

Dependencies point one way only. Scripts import the environment; the
environment imports nothing from scripts.
### Run order
```
1. smoke_test.py           verify the env doesn't crash     (seconds)
2. check_reset_pose.py     verify the start pose            (seconds)
3. train_base_policy.py    produce models/base_policy.zip   (30-60 min)
4. evaluate_policy.py      watch it — does it look alive?   (seconds)
5. diagnose_gait.py        prove numerically it's walking   (seconds)
6. baseline_fault_eval.py  produce Baseline A data          (minutes)
```

Steps 1-2 are cheap and catch expensive mistakes. Never skip them.