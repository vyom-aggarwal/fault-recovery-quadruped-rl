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
