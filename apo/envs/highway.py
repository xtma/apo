from rlpyt.envs.gym import GymEnvWrapper
import gym
from gym.wrappers.time_limit import TimeLimit


def make_highway_env(id):
    import highway_env
    if id == "lane-keeping-v0":  # return lane-keeping-v0 directly
        return GymEnvWrapper(gym.make(id))
    env = gym.make(id, config={
        "action": {
            "type": "ContinuousAction"
        },
        'simulation_frequency': 5,
    })
    env = TimeLimit(env, max_episode_steps=env.config['duration'])
    env = GymEnvWrapper(env)
    return env
