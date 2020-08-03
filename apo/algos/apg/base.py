from collections import namedtuple

from apo.algos.utils import discount_return, generalized_advantage_estimation
from rlpyt.algos.base import RlAlgorithm
from rlpyt.algos.utils import valid_from_done

# Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "entropy", "perplexity", "eta", "valueBias"])
AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])


class AveragePolicyGradientAlgo(RlAlgorithm):
    """
    Base policy gradient / actor-critic algorithm, which includes
    initialization procedure and processing of data samples to compute
    advantages.
    """

    bootstrap_value = True  # Tells the sampler it needs Value(State')
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False, examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls(agent.parameters(), lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)

        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset

    def update_eta(self, samples):
        """
        Update the estimated average performance with new samples.
        """
        reward, value = (samples.env.reward, samples.agent.agent_info.value)
        reward_mean, value_mean = reward.mean().item(), value.mean().item()
        if self.eta is None:
            self.eta = reward_mean
        else:
            self.eta = (1 - self.lr_eta) * self.eta + self.lr_eta * reward_mean
        if self.value_bias is None:
            self.value_bias = value_mean
        else:
            self.value_bias = (1 - self.lr_eta) * self.value_bias + self.lr_eta * value_mean
        return reward_mean, value_mean

    def process_returns(self, samples):
        """
        Compute bootstrapped returns and advantages from a minibatch of
        samples.  Uses either discounted returns (if ``self.gae_lambda==1``)
        or generalized advantage estimation.  Mask out invalid samples
        according to ``mid_batch_reset`` or for recurrent agent.  Optionally,
        normalize advantages.
        """
        reward, done, value, bv = (samples.env.reward, samples.env.done, samples.agent.agent_info.value,
                                   samples.agent.bootstrap_value)
        done = done.type(reward.dtype)

        if self.gae_lambda == 1:  # GAE reduces to MC.
            return_ = discount_return(reward, done, bv, self.eta)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(reward, value, done, bv, self.eta, self.gae_lambda)

        return_ -= self.rm_vbias_coeff * self.value_bias

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
        else:
            valid = None  # OR torch.ones_like(done)

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        return return_, advantage, valid
