import torch

from apo.algos.apg.base import AveragePolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_method
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import valid_mean


class AAC(AveragePolicyGradientAlgo):
    """
    Advantage Actor Critic algorithm (synchronous).  Trains the agent by
    taking one gradient step on each iteration of samples, with advantages
    computed by generalized advantage estimation.
    """

    def __init__(
        self,
        longrun=True,
        learning_rate=0.001,
        lr_eta=0.1,  # learning rate of average performance
        rm_vbias_coeff=0.1,  # remove value bias
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        OptimCls=torch.optim.Adam,
        optim_kwargs=None,
        clip_grad_norm=1.,
        initial_optim_state_dict=None,
        discount=1.,
        gae_lambda=0.9,
        normalize_advantage=False,
        bootstrap_timelimit=True,
    ):
        """Saves the input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.eta = None  # initial estimation of average performance
        self.value_bias = None  # initial estimation of average performance
        if not self.longrun:  # don't consider the long-run tricks
            self.eta = 0
            self.value_bias = 0
            self.lr_eta = 0
            self.rm_vbias_coeff = 0

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size  # For logging.

    def optimize_agent(self, itr, samples):
        """
        Train the agent on input samples, by one gradient step.
        """
        if hasattr(self.agent, "update_obs_rms"):
            # NOTE: suboptimal--obs sent to device here and in agent(*inputs).
            self.agent.update_obs_rms(samples.env.observation)
        eta, valueBias = self.update_eta(samples)
        self.optimizer.zero_grad()
        loss, entropy, perplexity = self.loss(samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info = OptInfo(
            loss=loss.item(),
            gradNorm=grad_norm.item(),
            entropy=entropy.item(),
            perplexity=perplexity.item(),
            eta=eta,
            valueBias=valueBias,
        )
        self.update_counter += 1
        return opt_info

    def loss(self, samples):
        """
        Computes the training loss: policy_loss + value_loss + entropy_loss.
        Policy loss: log-likelihood of actions * advantages
        Value loss: 0.5 * (estimated_value - return) ^ 2
        Organizes agent inputs from training samples, calls the agent instance
        to run forward pass on training data, and uses the
        ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        if self.agent.recurrent:
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T = 0.
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        # TODO: try to compute everyone on device.
        return_, advantage, valid = self.process_returns(samples)

        dist = self.agent.distribution
        logli = dist.log_likelihood(samples.agent.action, dist_info)
        pi_loss = -valid_mean(logli * advantage, valid)

        value_error = 0.5 * (value - return_)**2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)

        return loss, entropy, perplexity
