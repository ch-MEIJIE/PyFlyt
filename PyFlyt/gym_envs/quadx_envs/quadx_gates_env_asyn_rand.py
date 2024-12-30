"""QuadX Gates Environment."""
from __future__ import annotations

import copy
import math
import os
from typing import Any, Literal, Union

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from scipy.stats import norm

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv

ACTIONS = {
    "0": [0.0, 1.0, 0.0, 0.0],   # Forward
    "1": [0.0, -1.0, 0.0, 0.0],  # Backward
    "2": [1.0, 0.0, 0.0, 0.0],   # Right
    "3": [-1.0, 0.0, 0.0, 0.0],  # Left
    # "4": [0.0, 0.0, 1.0, 0.0],   # Up
    # "5": [0.0, 0.0, -1.0, 0.0],  # Down
    "4": [0.0, 0.0, 0.0, -1.0],   # CW
    "5": [0.0, 0.0, 0.0, 1.0],  # CCW
    "6": [0.0, 0.0, 0.0, 0.0],   # Hover
    "7": [0.0, 0.0, 0.0, 0.0],   # Keep
}


class BCIsimulator():
    def __init__(self, accuracy=0.99, overlap_time=1):
        self.accuracy = accuracy
        self.overlap_time = overlap_time
        self.threshold = (overlap_time/2.0)\
            - (norm.ppf(accuracy)*(overlap_time/12)**0.5)
        action_len = len(ACTIONS)
        self.decision_vec = np.zeros((overlap_time, action_len))
        self.ptr = 0
        self.one_hot_mat = np.eye(action_len)
        self.last_velocity_vec = np.zeros((4,))

    def encode(self, action):
        self.ptr += 1
        random_prob = np.random.uniform(0, 1, 1)
        self.decision_vec[self.ptr % self.overlap_time] \
            = self.one_hot_mat[action]*random_prob

    def decode(self):
        if self.ptr % self.overlap_time != 0 or self.ptr == 0:
            return None
        else:
            decision = np.argmax(np.sum(self.decision_vec, axis=0))
            decision_confidence = np.max(np.sum(self.decision_vec, axis=0))
            if decision_confidence < self.threshold:
                # random select a decision from 0~9 expect decision
                decision = np.random.choice(
                    [i for i in range(8) if i != decision])
                velocity_vec = self.map_decision(decision)
            else:
                velocity_vec = self.map_decision(decision)
            self.last_velocity_vec = velocity_vec
            return velocity_vec

    def map_decision(self, decision):
        if decision == 7:
            return self.last_velocity_vec
        elif decision == 6:
            return list(np.zeros((4,)))
        else:
            return ACTIONS[str(decision)]

    def reset(self):
        self.ptr = 0
        self.decision_vec = np.zeros((self.overlap_time, len(ACTIONS)))
        self.last_velocity_vec = np.zeros((4,))


class QuadXUVRZGatesRandEnv(QuadXBaseEnv):
    """QuadX Gates Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust

    The target is a set of `[x, y, z, yaw]` targets in space

    Reward is -(distance from waypoint + angle error) for each timestep,
    and -100.0 for hitting the ground.

    Args:
        flight_mode (int): the flight mode of the UAV
        num_targets (int): num_targets
        goal_reach_distance (float): goal_reach_distance
        min_gate_height (float): min_gate_height
        max_gate_angles (list[float]): max_gate_angles
        min_gate_distance (float): min_gate_distance
        max_gate_distance (float): max_gate_distance
        camera_resolution (tuple[int, int]): camera_resolution
        max_duration_seconds (float): max_duration_seconds
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution
    """

    def __init__(
        self,
        targets_num: int = 3,
        flight_mode: int = 4,
        bci_accuracy: float = 0.99,
        goal_reach_distance: float = 0.21,
        gate_height: float = 1.5,
        max_gate_angles: list[float] = [0.0, 0.0, 1.0],
        min_gate_distance: float = 1.0,
        max_gate_distance: float = 2.0,
        camera_resolution: tuple[int, int] | None = (256, 144),
        max_duration_seconds: float = 120.0,
        angle_representation: Literal["euler", "quaternion"] = "euler",
        agent_hz: int = 10,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        action_overlap: int = 4,
        seed: None | int = None,
        asyn: bool = True,
    ):
        """__init__.

        Args:
            flight_mode (int): the flight mode of the UAV
            num_targets (int): num_targets
            goal_reach_distance (float): goal_reach_distance
            min_gate_height (float): min_gate_height
            max_gate_angles (list[float]): max_gate_angles
            min_gate_distance (float): min_gate_distance
            max_gate_distance (float): max_gate_distance
            camera_resolution (tuple[int, int]): camera_resolution
            max_duration_seconds (float): max_duration_seconds
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution
        """
        super().__init__(
            flight_mode=flight_mode,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        self.asyn = asyn
        self.targets_num = targets_num
        if self.targets_num == 0:
            self.free_mode = True
        else:
            self.free_mode = False

        if self.asyn and not self.free_mode:
            # 9 action space for a watch video action
            self.action_space = spaces.Discrete(9)
            self.combined_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.attitude_space.shape[0]
                       + 5  # Action shape(1)+velocity vector shape(4)
                       + self.auxiliary_space.shape[0],
                       ),
                dtype=np.float64,
            )
            self.observation_space = spaces.Dict(
                {
                    "attitude": self.combined_space,
                    "target_deltas": spaces.Sequence(
                        space=spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float64,
                        ),
                        stack=True,
                    ),
                    "target_delta_bound": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(6,),
                        dtype=np.float64,
                    ),
                    "updated": spaces.Discrete(2),
                }
            )
        elif not self.asyn and not self.free_mode:
            self.action_space = spaces.Discrete(8)
            self.combined_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.attitude_space.shape[0]
                       + 5  # Action shape(1)+velocity vector shape(4)
                       + self.auxiliary_space.shape[0],
                       ),
                dtype=np.float64,
            )
            self.observation_space = spaces.Dict(
                {
                    "attitude": self.combined_space,
                    "target_deltas": spaces.Sequence(
                        space=spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(3,),
                            dtype=np.float64,
                        ),
                        stack=True,
                    ),
                    "target_delta_bound": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(6,),
                        dtype=np.float64,
                    ),
                }
            )
        elif self.free_mode:
            self.action_space = spaces.Discrete(8)
            self.combined_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.attitude_space.shape[0]
                       + 5  # Action shape(1)+velocity vector shape(4)
                       + self.auxiliary_space.shape[0],
                       ),
                dtype=np.float64,
            )
            self.observation_space = spaces.Dict(
                {
                    "attitude": self.combined_space,
                }
            )
        else:
            raise ValueError("Invalid mode")

        """ ENVIRONMENT CONSTANTS """
        self.bci = BCIsimulator(accuracy=bci_accuracy)
        self.velocity_buffer = np.zeros((action_overlap, 4))
        self.action_overlap = action_overlap
        self.step_ptr = 0

        """Target related parameters"""
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.min_gate_height = gate_height
        self.goal_reach_distance = goal_reach_distance
        self.gate_obj_dir = os.path.join(file_dir, "../models/race_gate.urdf")
        self.max_gate_distance = max_gate_distance
        self.min_gate_distance = min_gate_distance
        self.max_gate_angles = np.array([max_gate_angles])
        self.targets_right_bound = list()
        self.targets_left_bound = list()
        self.target_reached_count = 0

        """Camera related parameters"""
        self.camera_resolution = camera_resolution

        """System related parameters"""
        self.seed = seed
        if self.asyn:
            self.frozen_obs = dict()
            self.update = 1

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict, dict]:
        """Resets the environment.

        Args:
            seed: seed to pass to the base environment.
            options: None
        """
        if seed is None:
            seed = self.seed
        aviary_options = dict()
        if self.camera_resolution is not None:
            aviary_options["use_camera"] = True
            aviary_options["use_gimbal"] = False
            aviary_options["camera_resolution"] = self.camera_resolution
            aviary_options["camera_angle_degrees"] = 15.0
        else:
            aviary_options["use_camera"] = False

        super().begin_reset(seed, options, aviary_options)
        self.action = 0
        self.velocity_vec = np.zeros((4,))
        self.step_ptr = 0
        self.bci.reset()
        self.velocity_buffer = np.zeros((self.action_overlap, 4))
        self.load_senario("samurai.urdf")

        """GATES GENERATION"""
        if not self.free_mode:
            self.gates = []
            self.targets = []
            self.targets_right_bound = []
            self.targets_left_bound = []
            self.generate_gates()
            self.target_reached_count = 0
            self.info["current_target"] = self.targets[0]

        super().end_reset(seed, options)

        if self.asyn:
            # save the reset state to frozen_obs
            self.frozen_obs = self.state

            # return the self.frozen_obs and info
            return self.frozen_obs, self.info
        else:
            return self.state, self.info

    def load_senario(self, senario: str = "stadium.sdf") -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(senario)

    def generate_gates(self) -> None:
        """Generates the gates."""
        # sample a bunch of distances for gate distances
        distences = np.random.uniform(
            self.min_gate_distance, self.max_gate_distance, size=(self.targets_num, ))
        angles = np.random.uniform(
            -0.7, 0.7, size=(self.targets_num, 3))
        angles = np.multiply(angles, self.max_gate_angles)

        # starting position and angle
        gate_pos = np.array([0.0, 0.0, 1.0])
        gate_ang = np.array([0.0, 0.0, 0.0])

        for new_dist, new_ang in zip(distences, angles):

            # old rotation matrix and quaternion
            old_quat = p.getQuaternionFromEuler(gate_ang)
            old_mat = np.array(
                p.getMatrixFromQuaternion(old_quat)).reshape(3, 3)

            # new rotation matrix and quaternion
            new_quat = p.getQuaternionFromEuler(new_ang)
            new_mat = np.array(
                p.getMatrixFromQuaternion(new_quat)).reshape(3, 3)

            # rotate new distance by old angle then new angle
            new_dist = np.array([new_dist, 0.0, 0.0])
            new_dist = new_mat @ old_mat @ new_dist

            gate_pos += new_dist
            gate_ang += new_ang

            gate_quat = p.getQuaternionFromEuler(gate_ang)

            # store the new target and gates
            self.targets.append(copy.copy(gate_pos))

            # compute edge of the gate
            gate_right_bound = np.array([0.0, 0.0, 1.0])
            gate_left_bound = np.array([0.0, 0.0, 1.0])
            # left bound x
            if gate_ang[2] > 0:
                gate_left_bound[0] = gate_pos[0] - 0.25*math.sin(gate_ang[2])
                gate_left_bound[1] = gate_pos[1] + 0.25*math.cos(gate_ang[2])
                gate_right_bound[0] = gate_pos[0] + 0.25*math.sin(gate_ang[2])
                gate_right_bound[1] = gate_pos[1] - 0.25*math.cos(gate_ang[2])
            else:
                gate_left_bound[0] = gate_pos[0] + 0.25*math.sin(gate_ang[2])
                gate_left_bound[1] = gate_pos[1] - 0.25*math.cos(gate_ang[2])
                gate_right_bound[0] = gate_pos[0] - 0.25*math.sin(gate_ang[2])
                gate_right_bound[1] = gate_pos[1] + 0.25*math.cos(gate_ang[2])

            # store the bounds
            self.targets_right_bound.append(copy.copy(gate_right_bound))
            self.targets_left_bound.append(copy.copy(gate_left_bound))

            self.gates.append(
                self.env.loadURDF(
                    self.gate_obj_dir,
                    basePosition=gate_pos,
                    baseOrientation=gate_quat,
                    useFixedBase=True,
                )
            )

        # colour the first gate
        self.colour_first_gate()
        self.colour_other_gate()

    def get_current_gate(self):
        return self.targets[0], self.targets_right_bound[0], self.targets_left_bound[0]

    def colour_dead_gate(self, gate: int) -> None:
        """Colours the gates that are done.

        Args:
            gate (int): body ID of the gate
        """
        # colour the dead gates red
        for i in range(p.getNumJoints(gate)):
            p.changeVisualShape(
                gate,
                linkIndex=i,
                rgbaColor=(1, 0, 0, 1),
            )

    def colour_first_gate(self) -> None:
        """Colours the immediate target gate."""
        # colour the first gate green
        for i in range(p.getNumJoints(self.gates[0])):
            p.changeVisualShape(
                self.gates[0],
                linkIndex=i,
                rgbaColor=(0, 1, 0, 1),
            )

    def colour_other_gate(self) -> None:
        """Colours gates that are neither targets nor dead."""
        # colour all other gates yellow
        for gate in self.gates[1:]:
            for i in range(p.getNumJoints(gate)):
                p.changeVisualShape(
                    gate,
                    linkIndex=i,
                    rgbaColor=(1, 1, 0, 1),
                )

    def compute_state(self) -> None:
        """This returns the observation as well as the distances to target.

        - "attitude" (Box)
            - ang_vel (vector of 3 values)
            - ang_pos (vector of 3/4 values)
            - lin_vel (vector of 3 values)
            - lin_pos (vector of 3 values)
            - previous_action (vector of 4 values)
            - auxiliary information (vector of 4 values)
        - "target_deltas" (Graph)
            - list of body_frame distances to target (vector of 3/4 values)
        """
        ang_vel, ang_pos, lin_vel, lin_pos, quarternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # rotation matrix
        rotation = np.array(p.getMatrixFromQuaternion(
            quarternion)).reshape(3, 3).T

        if not self.free_mode:
            # drone to target
            target_deltas = np.matmul(rotation, (self.targets - lin_pos).T).T
            self.dis_error_scalar = np.linalg.norm(target_deltas[0])

            # drone to the next gate's edge
            target_delta_bound = np.zeros((6,))
            if len(self.gates) > 0:
                target_delta_bound[0:3] = np.matmul(
                    rotation,
                    (self.targets_right_bound[0] - lin_pos).T).T
                target_delta_bound[3:6] = np.matmul(
                    rotation,
                    (self.targets_left_bound[0] - lin_pos).T).T

        # combine everything
        new_state = dict()
        if self.angle_representation == 0:
            # 0:3 ang_vel, 3:6 euler angle, 6:9 lin_vel, 9:12 lin_pos
            # 12 action, 13:17 velocity_vec, 17:21 aux_state
            new_state["attitude"] = np.array(
                [*ang_vel,
                 *ang_pos,
                 *lin_vel,
                 *lin_pos,
                 self.action,
                 *self.velocity_vec,
                 *aux_state]
            )
        elif self.angle_representation == 1:
            # 0:3 ang_vel, 3:7 quarternion, 7:10 lin_vel, 10:13 lin_pos
            # 13 action, 14:18 velocity_vec, 18:22 aux_state
            new_state["attitude"] = np.array(
                [*ang_vel,
                 *quarternion,
                 *lin_vel,
                 *lin_pos,
                 self.action,
                 *self.velocity_vec,
                 *aux_state]
            )

        if not self.free_mode:
            # distances to targets
            new_state["target_deltas"] = target_deltas
            new_state["target_delta_bound"] = target_delta_bound
        if self.asyn and not self.free_mode:
            new_state["updated"] = self.update

        self.state = new_state

    def step(self, action: np.ndarray) -> tuple[
            Any, Union[float, Any], bool, bool, dict[str, Any]]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info
        """
        # unsqueeze the action to be usable in aviary

        self.action = action.copy()
        if self.asyn and self.action == 8:
            fake_action = np.random.choice([i for i in range(8)])
            self.bci.encode(fake_action)
            velocity_mapped = self.bci.decode()
            self.update = 1
        else:
            self.bci.encode(self.action)
            velocity_mapped = self.bci.decode()
            self.update = 0
        if velocity_mapped is not None:
            self.velocity_buffer[self.step_ptr %
                                 self.action_overlap] = velocity_mapped
            self.step_ptr += 1
        sum_velocity =\
            np.sum(self.velocity_buffer, axis=0, keepdims=False)[
                [1, 0, 3, 2]] * 0.05
        sum_velocity[-1] = 1.0
        self.velocity_vec = sum_velocity

        # reset the reward and set the action
        self.reward = -0.01*self.step_count
        self.env.set_setpoint(0, setpoint=self.velocity_vec)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        if self.asyn and self.action == 8 and not self.free_mode:
            # update the frozen_obs with the new state
            self.frozen_obs = self.state
        elif self.asyn and not self.free_mode:
            # update the action in the frozen_obs
            self.frozen_obs['attitude'][13] = self.action
            # make the update flag to 0
            self.frozen_obs['updated'] = np.int64(0)

        # distance reward
        if not self.free_mode:
            self.reward += 1.0/(self.dis_error_scalar+1)
        # increment step count
        self.step_count += 1

        if self.asyn:
            return self.frozen_obs, self.reward, self.termination, self.truncation, self.info
        else:
            return self.state, self.reward, self.termination, self.truncation, self.info

    @property
    def target_reached(self) -> bool:
        """Checks if the immediate target has been reached."""
        if self.dis_error_scalar < self.goal_reach_distance:
            return True
        else:
            return False

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward of the current step."""
        super().compute_base_term_trunc_reward()
        if not self.free_mode:
            # out of range of next gate
            if self.dis_error_scalar > 2 * self.max_gate_distance:
                self.reward += -100.0
                self.info["out_of_bounds"] = True
                self.termination = self.termination or True

            # target reached
            if self.target_reached:
                self.target_reached_count += 1
                # print(f"Target reached: {self.target_reached_count}")
                self.reward += 100.0
                if len(self.targets) > 1:
                    # still have targets to go
                    self.targets = self.targets[1:]
                    self.info["current_target"] = self.targets[0]
                    self.targets_left_bound = self.targets_left_bound[1:]
                    self.targets_right_bound = self.targets_right_bound[1:]
                else:
                    self.reward += 500.0
                    self.info["env_complete"] = True
                    self.termination = self.termination or True

                # shift the gates and recolour the reached one
                self.colour_dead_gate(self.gates[0])
                if len(self.gates) > 1:
                    self.gates = self.gates[1:]
                    # colour the new target
                    self.colour_first_gate()
