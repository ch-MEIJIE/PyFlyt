"""Registers PyFlyt environments into Gymnasium."""
from gymnasium.envs.registration import register

from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv

# QuadX Envs
register(
    id="PyFlyt/QuadX-Hover-v2",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_hover_env:QuadXHoverEnv",
)
register(
    id="PyFlyt/QuadX-Waypoints-v2",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env:QuadXWaypointsEnv",
)
register(
    id="PyFlyt/QuadX-Gates-v2",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_gates_env:QuadXGatesEnv",
)

register(
    id="PyFlyt/QuadX-Velocity_Gates-v2",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_velocity_gates_env:QuadXVelocityGatesEnv",

)

register(
    id="PyFlyt/QuadX-UVRZ-Gates-v2",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_velocity_gates_env_uvvrz:QuadXUVRZGatesEnv",
)

register(
    id="PyFlyt/QuadX-UVRZ-Render-Gates-v1",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_gates_env_uvvrz_render:QuadXUVRZGatesRenderEnv",
)

# Fixedwing Envs
register(
    id="PyFlyt/Fixedwing-Waypoints-v2",
    entry_point="PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env:FixedwingWaypointsEnv",
)

# Rocket Envs
register(
    id="PyFlyt/Rocket-Landing-v2",
    entry_point="PyFlyt.gym_envs.rocket_envs.rocket_landing_env:RocketLandingEnv",
)
