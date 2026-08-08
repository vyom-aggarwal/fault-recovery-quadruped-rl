import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
 
 
class QuadrupedFaultEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
 
    def __init__(
        self,
        render: bool = False,
        target_velocity: float = 0.5,
        max_episode_steps: int = 1000,
        action_repeat: int = 4,
    ):
        super().__init__()
        self.render_mode = "human" if render else None
        self.target_velocity = target_velocity
        self.max_episode_steps = max_episode_steps
        self.action_repeat = action_repeat
 
        self._client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
 
        self._standing_pose = [0.0, -0.7, 0.7] * 4
        self._action_scale = 0.5 
 
        self.robot_id = None
        self.plane_id = None
        self.joint_ids = []
        self.num_joints = 0
        self.step_count = 0
 
        self.active_fault = None
        self._torque_scale = None
        self._joint_lock_angle = {}
        self._sensor_noise_std = 0.0
        self._sensor_bias = None
        self._sensor_dropout_mask = None
        self._actuation_delay_steps = 0
        self._action_buffer = []
 
        # 12 joint angles + 12 joint velocities + 4 base orientation (quaternion)
        # + 3 base linear velocity + 3 base angular velocity = 34
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 12 + 12 + 4 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
 
    # Core Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)
 
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client)
        start_pos = [0, 0, 0.48]
        start_orn = [0, 0.5, 0.5, 0]
        self.robot_id = p.loadURDF(
            "laikago/laikago.urdf",
            start_pos,
            start_orn,
            physicsClientId=self._client,
            flags=p.URDF_USE_SELF_COLLISION,
        )
 
        self.joint_ids = [
            j for j in range(p.getNumJoints(self.robot_id, physicsClientId=self._client))
            if p.getJointInfo(self.robot_id, j, physicsClientId=self._client)[2] == p.JOINT_REVOLUTE
        ]
        self.num_joints = len(self.joint_ids)
 
        # Snap directly to the standing pose (not zero) before letting physics run, so the robot starts from a valid stance rather than fighting its way there.
        for idx, joint_id in enumerate(self.joint_ids):
            p.resetJointState(self.robot_id, joint_id, self._standing_pose[idx], physicsClientId=self._client)
 
        # reset fault state every episode unless caller re-applies one
        self.active_fault = None
        self._torque_scale = np.ones(self.num_joints, dtype=np.float32)
        self._joint_lock_angle = {}
        self._sensor_noise_std = 0.0
        self._sensor_bias = np.zeros(self.num_joints, dtype=np.float32)
        self._sensor_dropout_mask = np.zeros(self.num_joints, dtype=bool)
        self._actuation_delay_steps = 0
        self._action_buffer = []
 
        # settle onto the standing pose under active position control (not free-fall)
        for _ in range(60):
            for idx, joint_id in enumerate(self.joint_ids):
                p.setJointMotorControl2(
                    self.robot_id, joint_id, p.POSITION_CONTROL,
                    targetPosition=self._standing_pose[idx], force=20.0,
                    physicsClientId=self._client,
                )
            p.stepSimulation(physicsClientId=self._client)
 
        self.step_count = 0
        obs = self._get_obs()
        return obs, {}
 
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
 
        self._action_buffer.append(action)
        delay = self._actuation_delay_steps
        if len(self._action_buffer) > delay:
            applied_action = self._action_buffer.pop(0)
        else:
            applied_action = np.zeros_like(action)
 
        applied_action = self._apply_fault_to_action(applied_action)
 
        max_torque = 20.0
        for _ in range(self.action_repeat):
            for idx, joint_id in enumerate(self.joint_ids):
                if joint_id in self._joint_lock_angle:
                    # joint lock fault
                    target = self._joint_lock_angle[joint_id]
                else:
                    target = self._standing_pose[idx] + float(applied_action[idx]) * self._action_scale
 
                torque_limit = max_torque * self._torque_scale[idx]
                p.setJointMotorControl2(
                    self.robot_id,
                    joint_id,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=torque_limit,
                    physicsClientId=self._client,
                )
            p.stepSimulation(physicsClientId=self._client)
 
        self.step_count += 1
        obs = self._get_obs()
        reward, info = self._compute_reward()
        terminated = self._check_fallen()
        if terminated:
            reward -= 10.0  # explicit fall penalty
        truncated = self.step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, info
 
    def close(self):
        if p.isConnected(physicsClientId=self._client):
            p.disconnect(physicsClientId=self._client)
 
    # Fault injection API — call mid-episode to test recovery
    def trigger_fault(self, fault_type: str, joint: int = None, severity: float = 1.0):
        """
        fault_type: one of
            "torque_limit"   - caps target joint's torque to `severity` fraction of nominal
            "joint_lock"     - freezes target joint at its current angle
            "actuation_delay"- adds `severity` (int) steps of command lag, all joints
            "sensor_dropout" - target joint's angle reading intermittently drops to 0
            "sensor_noise"   - adds gaussian noise (std=severity) to all joint readings
        joint: joint index (required for torque_limit, joint_lock, sensor_dropout)
        severity: fault-specific magnitude (see above)
        """
        if joint is None and fault_type in ("torque_limit", "joint_lock", "sensor_dropout"):
            joint = self.np_random.integers(0, self.num_joints)
 
        if fault_type == "torque_limit":
            self._torque_scale[joint] = severity
        elif fault_type == "joint_lock":
            current_angle = p.getJointState(self.robot_id, self.joint_ids[joint], physicsClientId=self._client)[0]
            self._joint_lock_angle[self.joint_ids[joint]] = current_angle
        elif fault_type == "actuation_delay":
            self._actuation_delay_steps = int(severity)
        elif fault_type == "sensor_dropout":
            self._sensor_dropout_mask[joint] = True
        elif fault_type == "sensor_noise":
            self._sensor_noise_std = severity
        else:
            raise ValueError(f"Unknown fault_type: {fault_type}")
 
        self.active_fault = {"type": fault_type, "joint": joint, "severity": severity}
 
    def clear_faults(self):
        self._torque_scale[:] = 1.0
        self._joint_lock_angle = {}
        self._sensor_noise_std = 0.0
        self._sensor_bias[:] = 0.0
        self._sensor_dropout_mask[:] = False
        self._actuation_delay_steps = 0
        self.active_fault = None
 
    def _apply_fault_to_action(self, action):
        # placeholder hook 
        return action
 
    # Observation / reward / termination
    def _get_obs(self):
        joint_angles = np.zeros(self.num_joints, dtype=np.float32)
        joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        for idx, joint_id in enumerate(self.joint_ids):
            angle, vel, _, _ = p.getJointState(self.robot_id, joint_id, physicsClientId=self._client)
            joint_angles[idx] = angle
            joint_velocities[idx] = vel

        if self._sensor_noise_std > 0:
            joint_angles = joint_angles + self.np_random.normal(0, self._sensor_noise_std, size=joint_angles.shape)
        joint_angles = joint_angles + self._sensor_bias
        joint_angles[self._sensor_dropout_mask] = 0.0
 
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
 
        obs = np.concatenate([
            joint_angles,
            joint_velocities,
            np.array(orn, dtype=np.float32),
            np.array(lin_vel, dtype=np.float32),
            np.array(ang_vel, dtype=np.float32),
        ]).astype(np.float32)
        return obs
 
    def _get_orientation_frame(self, orn):
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        up_axis_world = rot_matrix[:, 1]
        forward_axis_world = rot_matrix[:, 2]
        return up_axis_world, forward_axis_world
 
    def _compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(self.robot_id, physicsClientId=self._client)
        up_axis, forward_axis = self._get_orientation_frame(orn)
 
        forward_vel = float(np.dot(lin_vel, forward_axis))
        vel_reward = -abs(forward_vel - self.target_velocity)
 
        torque_penalty = 0.0
        max_t = getattr(self, "max_torque", 20.0)
        for joint_id in self.joint_ids:
            _, _, _, applied_torque = p.getJointState(self.robot_id, joint_id, physicsClientId=self._client)
            normalized_torque = applied_torque / max_t if max_t > 0 else 0.0
            torque_penalty += normalized_torque ** 2
        energy_penalty = -0.03 * torque_penalty
 
        upright_alignment = float(up_axis[2])
        stability_penalty = -1.0 * (1.0 - upright_alignment) ** 2
        height_penalty = -1.0 * max(0.0, 0.50 - pos[2])  # penalize crouching 
 
        alive_bonus = 1.0
 
        reward = vel_reward + energy_penalty + stability_penalty + height_penalty + alive_bonus
        info = {
            "forward_vel": forward_vel,
            "vel_reward": vel_reward,
            "energy_penalty": energy_penalty,
            "stability_penalty": stability_penalty,
            "height": pos[2],
            "upright_alignment": upright_alignment,
        }
        return reward, info
 
    def _check_fallen(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self._client)
        up_axis, _ = self._get_orientation_frame(orn)
        upright_alignment = up_axis[2]
        # fallen if too low or tipped more than ~60 degrees from vertical
        fallen = pos[2] < 0.35 or upright_alignment < 0.5
        return fallen