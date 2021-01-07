from rlpyt.agents.pg.mujoco import MujocoFfAgent, MujocoLstmAgent


class EvalMixin:
    """
    Set std = 0 in evaluation.
    """

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.distribution.set_std(None)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).


class MujocoFfEvalAgent(EvalMixin, MujocoFfAgent):
    pass


class MujocoLstmEvalAgent(EvalMixin, MujocoLstmAgent):
    pass
