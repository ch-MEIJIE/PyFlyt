"""Base Multiagent QuadX Environment for use with the Pettingzoo API."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Tuple

import numpy as np
from gymnasium import Space, spaces
from pettingzoo import AECEnv
import pybullet as p
from pettingzoo.utils import agent_selector

from PyFlyt.core import Aviary


class MAQuadXBaseEnv(AECEnv):
    """Base Multiagent QuadX Environment for use with the Pettingzoo API.

    Args:
        start_pos (np.ndarray): start_pos
        start_orn (np.ndarray): start_orn
        flight_dome_size (float): flight_dome_size
        max_duration_seconds (float): max_duration_seconds
        angle_representation (str): angle_representation
        agent_hz (int): agent_hz
        render_mode (None | str): render_mode
    """

    metadata = {"render_modes": ["human"], "name": "ma_quadx_hover"}

    def __init__(
        self,
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        flight_dome_size: float = np.inf,
        max_duration_seconds: float = 10.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 30,
        render_mode: None | str = None,
    ):
        """__init__.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (str): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | str): render_mode
        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise AssertionError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode is not None:
            assert (
                render_mode in self.metadata["render_modes"]
            ), f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
        self.render_mode = True if render_mode is not None

        """SPACES"""
        # attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise AssertionError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        # action space
        angular_rate_limit = np.pi
        thrust_limit = 0.8
        high = np.array(
            [
                angular_rate_limit,
                angular_rate_limit,
                angular_rate_limit,
                thrust_limit,
            ]
        )
        low = np.array(
            [
                -angular_rate_limit,
                -angular_rate_limit,
                -angular_rate_limit,
                0.0,
            ]
        )
        self.action_space = lambda _: spaces.Box(low=low, high=high, dtype=np.float64) # pyright: ignore

        # observation space
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.auxiliary_space.shape[0],  # pyright: ignore
                + self.action_space.shape[0]  # pyright: ignore
            ),
            dtype=np.float64,
        )

        """ENVIRONMENT CONSTANTS"""
        # check the start_pos shapes
        assert len(start_pos.shape) == 2, f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        assert start_pos.shape[-1] == 3, f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        assert start_pos.shape == start_orn.shape, f"Expected `start_pos` to be of shape [num_agents, 3], got {start_pos.shape}."
        self.start_pos = start_pos
        self.start_orn = start_orn

        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

        # select agents
        self.num_possible_agents = len(start_pos)
        self.possible_agents = ["uav_" + str(r) for r in range(self.num_possible_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        """RUNTIME PARAMETERS"""
        self.current_actions = np.zeros(
            (self.num_possible_agents, *self.action_space(None).shape)
        )  # pyright: ignore
        self.past_actions = np.zeros(
            (self.num_possible_agents, *self.action_space(None).shape)
        )  # pyright: ignore

    def observation_space(self, _):
        """observation_space.

        Args:
            _:
        """
        raise NotImplementedError

    def observe(self, agent: str):
        """observe.

        Args:
            agent:
        """
        agent_id = self.agent_name_mapping[agent]
        self.observe_by_id(agent_id)

    def observe_by_id(self, agent_id: int):
        raise NotImplementedError

    def close(self):
        """close."""
        if hasattr(self, "aviary"):
            self.aviary.disconnect()

    def begin_reset(self, seed=None, options=dict()):
        """The first half of the reset function."""

        # if we already have an env, disconnect from it
        if hasattr(self, "aviary"):
            self.aviary.disconnect()
        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        for agent in self.agents:
            self.infos[agent] = dict()
            self.infos[agent]["out_of_bounds"] = False
            self.infos[agent]["collision"] = False
            self.infos[agent]["env_complete"] = False

        # Our agent_selector utility allows easy cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        # disconnect and rebuilt the environment
        if hasattr(self, "aviary"):
            self.aviary.disconnect()
        self.aviary = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=bool(self.render_mode),
            seed=seed,
        )

    def end_reset(self, seed=None, options=dict()):
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.aviary.register_all_new_bodies()

        # set flight mode
        self.aviary.set_mode(0)

        # wait for env to stabilize
        for _ in range(10):
            self.aviary.step()

    def compute_auxiliary_by_id(self, agent_id: int):
        """This returns the auxiliary state form the drone."""
        return self.aviary.aux_state(agent_id)

    def compute_attitude_by_id(self, agent_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quarternion (vector of 4 values)
        """
        raw_state = self.aviary.state(agent_id)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quarternion angles
        quarternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quarternion

    def compute_base_term_trunc_reward_info_by_id(self, agent_id: int) -> Tuple[bool, bool, float, dict[str, Any]]:
        """compute_base_term_trunc_reward_by_id."""
        # initialize
        term = False
        trunc = False
        reward = 0.0
        info = dict()

        # exceed step count
        trunc |= self.step_count > self.max_steps

        # collision
        if np.any(self.aviary.contact_array[agent_id]):
            reward -= 100.0
            info["collision"] = True
            term |= True

        # exceed flight dome
        if np.linalg.norm(self.aviary.state(agent_id)[-1]) > self.flight_dome_size:
            reward -= 100.0
            info["out_of_bounds"] = True
            term |= True

        return term, trunc, reward, info

    def compute_term_trunc_reward_info_by_id(self, agent_id: int) -> Tuple[bool, bool, float, dict[str, Any]]:
        """compute_term_trunc_reward_info_by_id.

        Args:
            agent_id (int): agent_id

        Returns:
            Tuple[bool, bool, float, dict[str, Any]]:
        """
        raise NotImplementedError

    def step(self, action: np.ndarray):
        """step.

        Args:
            action (np.ndarray): action
        """
        agent = self.agent_selection

        # terminate if agent is dead
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        # set the actions, clear the agent's cumulative reards since it's seen it
        self.current_actions[self.agent_name_mapping[agent]] = action
        self._cumulative_rewards[agent] = 0

        # environment logic
        if not self._agent_selector.is_last():
            # don't do anything if all agents haven't acted
            self._clear_rewards()
        else:
            # collect reward if it is the last agent to act
            self.aviary.set_all_setpoints(self.current_actions)
            self.past_actions = deepcopy(self.current_actions)

            # step enough times for one RL step
            for _ in range(self.env_step_ratio):
                self.aviary.step()

                # update reward, term, trunc, for each agent
                for ag in self.agents:
                    ag_id = self.agent_name_mapping[ag]

                    # compute term trunc reward
                    term, trunc, rew, info = self.compute_term_trunc_reward_info_by_id(ag_id)
                    self.terminations[ag] |= term
                    self.truncations[ag] |= trunc
                    self.rewards[ag] += rew
                    self.infos[ag] = {**self.infos[ag], **info}

        # accumulate rewards and select next agent
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

