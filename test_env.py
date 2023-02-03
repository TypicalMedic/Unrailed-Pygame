import unrailed_env
from pettingzoo.test import api_test
from pettingzoo.test import test_save_obs
import gymnasium as gym


cycles = 1000
env = unrailed_env.env("human", max_cycles=cycles)

env.reset()
for agent in env.agent_iter(cycles):
    observation, reward, termination, truncation, info = env.last()
    action = None if termination or truncation else env.action_space(agent).sample()  # this is where you would insert your policy
    env.step(action)

env.close()

# test_save_obs(env)

# api_test(env, num_cycles=1000, verbose_progress=False)

