import gymnasium
import PyFlyt.gym_envs # noqa
import numpy as np
import time

env = gymnasium.make("PyFlyt/QuadX-UVRZ-Gates-v2", render_mode="human", num_targets=1,agent_hz=2)
obs = env.reset()

termination = False
truncation = False
step = 0
action_choices = [0]
action_ptr = 0
old_time = time.time()

while not termination or truncation:
    action = env.action_space.sample()
    # print(type(action))
    # if step % 100 == 0:
    #     timecount = time.time()
    #     print("Time cost: ", timecount - old_time)
    #     old_time = timecount
    #     action = np.int64(action_choices[action_ptr])
    #     action_ptr += 1
    #     print(action)
    observation, reward, termination, truncation, info = env.step(action=action)
    print("Velocity: ", observation['attitude'][10: 13])
    # print("Info: ", info)
    step += 1

    # observation, reward, termination, truncation, info = env.step(action=0)
    # print(observation, reward, termination, truncation, info)