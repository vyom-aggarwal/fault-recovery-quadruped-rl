"""
Diagnostic: what does the robot's pose look like immediately after reset(),
before any policy action is applied? This isolates whether a bad starting
pose is an environment bug (reset/settle logic) vs. something the trained
policy is actively doing afterward.
 
Run:
    python scripts/check_reset_pose.py
"""
import os
import sys
 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
from envs.quadruped_env import QuadrupedFaultEnv
import pybullet as p
 
env = QuadrupedFaultEnv(render=True)  # GUI window so you can also SEE this pose directly
obs, info = env.reset(seed=0)
 
# Default PyBullet camera is distant/elevated and hard to judge a pose from.
# Snap to a close, direct side-on view instead.
p.resetDebugVisualizerCamera(
    cameraDistance=0.9,
    cameraYaw=90,       # looking along the robot's side, not from a corner
    cameraPitch=-5,     # nearly eye-level, not looking down from above
    cameraTargetPosition=[0, 0, 0.25],
    physicsClientId=env._client,
)
 
pos, orn = p.getBasePositionAndOrientation(env.robot_id, physicsClientId=env._client)
roll, pitch, yaw = p.getEulerFromQuaternion(orn)
contacts = p.getContactPoints(env.robot_id, env.plane_id, physicsClientId=env._client)
 
print(f"Height:  {pos[2]:.3f}  (expected ~0.30 if standing correctly)")
print(f"Roll:    {roll:.3f} rad  ({roll * 57.3:.1f} deg)")
print(f"Pitch:   {pitch:.3f} rad ({pitch * 57.3:.1f} deg)  <- large POSITIVE/NEGATIVE pitch = tipped forward/back")
print(f"Ground contact points: {len(contacts)}")
 
input("Press Enter to close the window...")
env.close()
 