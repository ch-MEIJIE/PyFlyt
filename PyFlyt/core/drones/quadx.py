from __future__ import annotations

import math

import numpy as np
import yaml
from pybullet_utils import bullet_client

from ..abstractions.base_controller import CtrlClass
from ..abstractions.base_drone import DroneClass
from ..abstractions.camera import Camera
from ..abstractions.motors import Motors
from ..abstractions.pid import PID


class QuadX(DroneClass):
    """QuadX instance that handles everything about a quadrotor in the X configuration."""

    def __init__(
        self,
        p: bullet_client.BulletClient,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        ctrl_hz: int,
        physics_hz: int,
        drone_model: str = "cf2x",
        model_dir: None | str = None,
        use_camera: bool = False,
        use_gimbal: bool = False,
        camera_angle_degrees: int = 20,
        camera_FOV_degrees: int = 90,
        camera_resolution: tuple[int, int] = (128, 128),
        np_random: None | np.random.RandomState = None,
    ):
        """Creates a drone in the QuadX configuration and handles all relevant control and physics.

        Args:
            p (bullet_client.BulletClient): p
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            ctrl_hz (int): ctrl_hz
            physics_hz (int): physics_hz
            model_dir (None | str): model_dir
            drone_model (str): drone_model
            use_camera (bool): use_camera
            use_gimbal (bool): use_gimbal
            camera_angle_degrees (int): camera_angle_degrees
            camera_FOV_degrees (int): camera_FOV_degrees
            camera_resolution (tuple[int, int]): camera_resolution
            np_random (None | np.random.RandomState): np_random
        """
        super().__init__(
            p=p,
            start_pos=start_pos,
            start_orn=start_orn,
            ctrl_hz=ctrl_hz,
            physics_hz=physics_hz,
            model_dir=model_dir,
            drone_model=drone_model,
            np_random=np_random,
        )
        """
        DRONE CONTROL
            motor ids correspond to quadrotor X in PX4, using the ENU convention
            control commands are in the form of pitch-roll-yaw-thrust
        """

        # All the params for the drone
        with open(self.param_path, "rb") as f:
            # load all params from yaml
            all_params = yaml.safe_load(f)
            motor_params = all_params["motor_params"]
            drag_params = all_params["drag_params"]
            ctrl_params = all_params["control_params"]

            # motor thrust and torque constants
            motor_ids = [0, 1, 2, 3]
            thrust_coef = (
                np.array(
                    [
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, +1.0],
                    ]
                )
                * motor_params["thrust_coef"]
            )
            torque_coef = (
                np.array(
                    [
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, +1.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 0.0, -1.0],
                    ]
                )
                * motor_params["torque_coef"]
            )
            noise_ratio = np.array([1.0] * 4) * motor_params["noise_ratio"]
            max_rpm = np.array([1.0] * 4) * np.sqrt(
                (motor_params["thrust_to_weight"] * 9.81)
                / (4 * motor_params["thrust_coef"])
            )
            tau = np.array([1.0] * 4) * motor_params["tau"]
            self.motors = Motors(
                p=self.p,
                physics_period=self.physics_period,
                np_random=self.np_random,
                uav_id=self.Id,
                motor_ids=motor_ids,
                tau=tau,
                max_rpm=max_rpm,
                thrust_coef=thrust_coef,
                torque_coef=torque_coef,
                noise_ratio=noise_ratio,
            )

            # motor mapping from command to individual motors
            self.motor_map = np.array(
                [
                    [+1.0, -1.0, +1.0, +1.0],
                    [-1.0, +1.0, +1.0, +1.0],
                    [+1.0, +1.0, -1.0, +1.0],
                    [-1.0, -1.0, -1.0, +1.0],
                ]
            )

            # pseudo drag coef
            self.drag_coef_xyz = drag_params["drag_coef_xyz"]
            self.drag_coef_pqr = drag_params["drag_coef_pqr"]

            self.Kp_ang_vel = np.array(ctrl_params["ang_vel"]["kp"])
            self.Ki_ang_vel = np.array(ctrl_params["ang_vel"]["ki"])
            self.Kd_ang_vel = np.array(ctrl_params["ang_vel"]["kd"])
            self.lim_ang_vel = np.array(ctrl_params["ang_vel"]["lim"])

            self.Kp_ang_pos = np.array(ctrl_params["ang_pos"]["kp"])
            self.Ki_ang_pos = np.array(ctrl_params["ang_pos"]["ki"])
            self.Kd_ang_pos = np.array(ctrl_params["ang_pos"]["kd"])
            self.lim_ang_pos = np.array(ctrl_params["ang_pos"]["lim"])

            self.Kp_lin_vel = np.array(ctrl_params["lin_vel"]["kp"])
            self.Ki_lin_vel = np.array(ctrl_params["lin_vel"]["ki"])
            self.Kd_lin_vel = np.array(ctrl_params["lin_vel"]["kd"])
            self.lim_lin_vel = np.array(ctrl_params["lin_vel"]["lim"])

            # input: linear position command
            # outputs: linear velocity
            self.Kp_lin_pos = np.array(ctrl_params["lin_pos"]["kp"])
            self.Ki_lin_pos = np.array(ctrl_params["lin_pos"]["ki"])
            self.Kd_lin_pos = np.array(ctrl_params["lin_pos"]["kd"])
            self.lim_lin_pos = np.array(ctrl_params["lin_pos"]["lim"])

            # height controllers
            z_pos_PID = PID(
                ctrl_params["z_pos"]["kp"],
                ctrl_params["z_pos"]["ki"],
                ctrl_params["z_pos"]["kd"],
                ctrl_params["z_pos"]["lim"],
                self.ctrl_period,
            )
            z_vel_PID = PID(
                ctrl_params["z_vel"]["kp"],
                ctrl_params["z_vel"]["ki"],
                ctrl_params["z_vel"]["kd"],
                ctrl_params["z_vel"]["lim"],
                self.ctrl_period,
            )
            self.z_PIDs = [z_vel_PID, z_pos_PID]
            self.PIDs = []

        """ CAMERA """
        self.use_camera = use_camera
        if self.use_camera:
            self.camera = Camera(
                p=self.p,
                uav_id=self.Id,
                camera_id=0,
                use_gimbal=use_gimbal,
                camera_FOV_degrees=camera_FOV_degrees,
                camera_angle_degrees=camera_angle_degrees,
                camera_resolution=camera_resolution,
            )

        """ CUSTOM CONTROLLERS """
        # dictionary mapping of controller_id to controller objects
        self.registered_controllers = dict()
        self.instanced_controllers = dict()
        self.registered_base_modes = dict()

        self.reset()

    def reset(self):
        """reset."""
        self.set_mode(0)
        self.setpoint = np.zeros((4))
        self.pwm = np.zeros((4))

        self.p.resetBasePositionAndOrientation(self.Id, self.start_pos, self.start_orn)
        self.motors.reset()
        self.update_state()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()

    def set_mode(self, mode):
        """
        vp, vq, vr = angular velocities
        p, q, r = angular positions
        u, v, w = local linear velocities
        x, y, z = linear positions
        vx, vy, vz = ground linear velocities
        T = thrust

        sets the flight mode:
           -1 - m1, m2, m3, m4
            0 - vp, vq, vr, T
            1 - p, q, r, vz
            2 - vp, vq, vr, z
            3 - p, q, r, z
            4 - u, v, vr, z
            5 - u, v, vr, vz
            6 - vx, vy, vr, vz
            7 - x, y, r, z
        """
        if (mode < -1 or mode > 7) and mode not in self.registered_controllers.keys():
            raise ValueError(
                f"`mode` must be between -1 and 7 or be registered in {self.registered_controllers.keys()=}, got {mode}."
            )

        self.mode = mode

        # for custom modes
        if mode in self.registered_controllers.keys():
            self.instanced_controllers[mode] = self.registered_controllers[mode]()
            mode = self.registered_base_modes[mode]

        # mode -1 means no controller present
        if mode == -1:
            return

        # preset setpoints on mode change
        if mode == 0:
            # racing mode, thrust to 0
            self.setpoint = np.array([0.0, 0.0, 0.0, -1.0])
        elif mode in [1, 5, 6]:
            # anything with a vz component, set to 0 vz
            self.setpoint = np.array([0.0, 0.0, 0.0, 0.0])
        elif mode == 7:
            # position mode just hold position
            self.setpoint = np.array(
                [*self.state[-1, :2], self.state[1, -1], self.state[-1, -1]]
            )
        else:
            # everything else set to 0 except z component maintain
            self.setpoint = np.array([0.0, 0.0, 0.0, 0.0])
            self.setpoint[-1] = self.state[-1, -1]

        # instantiate PIDs
        if mode in [0, 2]:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID]
        elif mode in [1, 3]:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos,
                self.Ki_ang_pos,
                self.Kd_ang_pos,
                self.lim_ang_pos,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID]
        elif mode in [4, 5, 6]:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos[:2],
                self.Ki_ang_pos[:2],
                self.Kd_ang_pos[:2],
                self.lim_ang_pos[:2],
                self.ctrl_period,
            )
            lin_vel_PID = PID(
                self.Kp_lin_vel,
                self.Ki_lin_vel,
                self.Kd_lin_vel,
                self.lim_lin_vel,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID, lin_vel_PID]
        elif mode == 7:
            ang_vel_PID = PID(
                self.Kp_ang_vel,
                self.Ki_ang_vel,
                self.Kd_ang_vel,
                self.lim_ang_vel,
                self.ctrl_period,
            )
            ang_pos_PID = PID(
                self.Kp_ang_pos,
                self.Ki_ang_pos,
                self.Kd_ang_pos,
                self.lim_ang_pos,
                self.ctrl_period,
            )
            lin_vel_PID = PID(
                self.Kp_lin_vel,
                self.Ki_lin_vel,
                self.Kd_lin_vel,
                self.lim_lin_vel,
                self.ctrl_period,
            )
            lin_pos_PID = PID(
                self.Kp_lin_pos,
                self.Ki_lin_pos,
                self.Kd_lin_pos,
                self.lim_lin_pos,
                self.ctrl_period,
            )
            self.PIDs = [ang_vel_PID, ang_pos_PID, lin_vel_PID, lin_pos_PID]

        for controller in self.PIDs:
            controller.reset()

    def register_controller(
        self,
        controller_id: int,
        controller_constructor: type[CtrlClass],
        base_mode: int,
    ):
        """register_controller.

        Args:
            controller_id (int): controller_id
            controller_constructor (type[CtrlClass]): controller_constructor
            base_mode (int): base_mode
        """
        assert (
            controller_id > 7
        ), f"`controller_id` must be more than 7, currently {controller_id}"
        assert (
            base_mode >= -1 and base_mode <= 7
        ), f"`base_mode` must be within -1 and 7, currently {base_mode}."

        self.registered_controllers[controller_id] = controller_constructor
        self.registered_base_modes[controller_id] = base_mode

    def update_state(self):
        """ang_vel, ang_pos, lin_vel, lin_pos"""
        lin_pos, ang_pos = self.p.getBasePositionAndOrientation(self.Id)
        lin_vel, ang_vel = self.p.getBaseVelocity(self.Id)

        # express vels in local frame
        rotation = np.array(self.p.getMatrixFromQuaternion(ang_pos)).reshape(3, 3).T
        lin_vel = np.matmul(rotation, lin_vel)
        ang_vel = np.matmul(rotation, ang_vel)

        # ang_pos in euler form
        ang_pos = self.p.getEulerFromQuaternion(ang_pos)

        self.state = np.stack([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)

    def update_control(self):
        """runs through controllers"""
        # this is the thing we cascade down controllers
        a_output = self.setpoint[:3].copy()
        z_output = self.setpoint[-1].copy()
        mode = self.mode

        # custom controllers run first if any
        if self.mode in self.registered_controllers.keys():
            custom_output = self.instanced_controllers[self.mode].step(
                self.state, self.setpoint
            )
            assert custom_output.shape == (
                4,
            ), f"custom controller outputting wrong shape, expected (4, ) but got {custom_output.shape}."

            # splice things out to be passed along
            a_output = custom_output[:3].copy()
            z_output = custom_output[-1].copy()
            mode = self.registered_base_modes[self.mode]

        # controller -1 means just direct to motor pwm commands
        if mode == -1:
            self.pwm = np.array([*a_output, z_output])
            return

        # angle controllers
        if mode in [0, 2]:
            a_output = self.PIDs[0].step(self.state[0], a_output)
        elif mode in [1, 3]:
            a_output = self.PIDs[1].step(self.state[1], a_output)
            a_output = self.PIDs[0].step(self.state[0], a_output)
        elif mode in [4, 5]:
            a_output[:2] = self.PIDs[2].step(self.state[2][:2], a_output[:2])
            a_output[:2] = np.array([-a_output[1], a_output[0]])
            a_output[:2] = self.PIDs[1].step(self.state[1][:2], a_output[:2])
            a_output = self.PIDs[0].step(self.state[0], a_output)
        elif mode == 6:
            c = math.cos(self.state[1, -1])
            s = math.sin(self.state[1, -1])
            rot_mat = np.array([[c, -s], [s, c]]).T
            a_output[:2] = np.matmul(rot_mat, a_output[:2])

            a_output[:2] = self.PIDs[2].step(self.state[2][:2], a_output[:2])
            a_output[:2] = np.array([-a_output[1], a_output[0]])
            a_output[:2] = self.PIDs[1].step(self.state[1][:2], a_output[:2])
            a_output = self.PIDs[0].step(self.state[0], a_output)
        elif mode == 7:
            a_output[:2] = self.PIDs[3].step(self.state[3][:2], a_output[:2])

            c = math.cos(self.state[1, -1])
            s = math.sin(self.state[1, -1])
            rot_mat = np.array([[c, -s], [s, c]]).T
            a_output[:2] = np.matmul(rot_mat, a_output[:2])

            a_output[:2] = self.PIDs[2].step(self.state[2][:2], a_output[:2])
            a_output = np.array([-a_output[1], a_output[0], a_output[2]])
            a_output = self.PIDs[1].step(self.state[1], a_output)
            a_output = self.PIDs[0].step(self.state[0], a_output)

        # height controllers
        if mode == 0:
            z_output = np.clip(z_output, 0.0, 1.0)
        elif mode == 1 or mode == 5 or mode == 6:
            z_output = self.z_PIDs[0].step(self.state[2][-1], z_output)
            z_output = np.clip(z_output, 0, 1)
        elif mode == 2 or mode == 3 or mode == 4 or mode == 7:
            z_output = self.z_PIDs[1].step(self.state[3][-1], z_output)
            z_output = self.z_PIDs[0].step(self.state[2][-1], z_output)
            z_output = np.clip(z_output, 0, 1)

        # mix the commands according to motor mix
        cmd = np.array([*a_output, z_output])
        self.pwm = np.matmul(self.motor_map, cmd)

        # deal with motor saturations
        if (high := np.max(self.pwm)) > 1.0:
            self.pwm /= high
        if (low := np.min(self.pwm)) < 0.05:
            self.pwm += (1.0 - self.pwm) / (1.0 - low) * (0.05 - low)

    def update_drag(self):
        """adds drag to the model, this is not physically correct but only approximation"""
        drag_pqr = -self.drag_coef_pqr * (np.array(self.state[0]) ** 2)
        drag_xyz = -self.drag_coef_xyz * (np.array(self.state[2]) ** 2)

        # warning, the physics is funky for bounces
        if len(self.p.getContactPoints()) == 0:
            self.p.applyExternalForce(
                self.Id, -1, drag_xyz, [0.0, 0.0, 0.0], self.p.LINK_FRAME
            )
            self.p.applyExternalTorque(self.Id, -1, drag_pqr, self.p.LINK_FRAME)

    def update_physics(self):
        """update_physics."""
        self.update_state()
        self.motors.pwm2forces(self.pwm)
        self.update_drag()

    def update_avionics(self):
        """
        updates state and control
        """
        self.update_control()

        if self.use_camera:
            self.rgbaImg, self.depthImg, self.segImg = self.camera.capture_image()
