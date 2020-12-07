import numpy as np

from rlpyt.samplers.collections import TrajInfo


class AverageTrajInfo(TrajInfo):

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.Reward = self.Return / self.Length
