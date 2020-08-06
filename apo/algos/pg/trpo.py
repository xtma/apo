import torch

from apo.algos.pg.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer
from rlpyt.agents.base import AgentInputs
from rlpyt.algos.pg.base import OptInfo, PolicyGradientAlgo
from rlpyt.utils.buffer import buffer_method, buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import valid_mean

LossInputs = namedarraytuple("LossInputs", ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])


class TRPO(PolicyGradientAlgo):
    """
    Trust Region Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.
    Note that this implementation only support Feedforward model.
    """

    def __init__(
        self,
        discount=0.99,
        learning_rate=0.001,
        max_constraint_value=0.01,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        OptimCls=ConjugateGradientOptimizer,
        vOptimCls=torch.optim.Adam,
        optim_kwargs=None,
        initial_optim_state_dict=None,
        gae_lambda=1,
        minibatches=4,
        epochs=4,
        linear_lr_schedule=True,
        normalize_advantage=False,
    ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False, examples=None, world_size=1, rank=0):
        """
        Build the torch optimizer and store other input attributes. Params
        ``batch_spec`` and ``examples`` are unused.
        """
        self.optimizer = self.OptimCls([*agent.model.mu.parameters(), agent.model.log_std],
                                       max_constraint_value=self.max_constraint_value)
        self.v_optimizer = self.vOptimCls(agent.model.v.parameters(), lr=self.learning_rate)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset
        self.rank = rank
        self.world_size = world_size
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.

    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid = self.process_returns(samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        batch_size = T * B
        mb_size = batch_size // self.minibatches
        # update policy
        loss, entropy, perplexity = self._train_policy(loss_inputs)
        opt_info.loss.append(loss.item())
        opt_info.entropy.append(entropy.item())
        opt_info.perplexity.append(perplexity.item())
        # update value
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = idxs % T
                B_idxs = idxs // T
                self._train_value(loss_inputs[T_idxs, B_idxs])
                self.update_counter += 1

        return opt_info

    def _train_policy(self, loss_inputs, init_rnn_state=None):
        """Train the policy."""
        self.optimizer.zero_grad()
        loss, entropy, perplexity = self._compute_loss_with_adv(loss_inputs, init_rnn_state, return_opt_info=True)
        loss.backward()
        self.optimizer.step(f_loss=lambda: self._compute_loss_with_adv(loss_inputs, init_rnn_state),
                            f_constraint=lambda: self._compute_kl_constraint(loss_inputs, init_rnn_state))
        return loss, entropy, perplexity

    def _train_value(self, loss_inputs, init_rnn_state=None):
        """Train the value."""
        agent_inputs, _, return_, _, valid, _ = loss_inputs
        _, value = self._process_batch(agent_inputs, init_rnn_state)
        self.v_optimizer.zero_grad()
        value_error = 0.5 * (value - return_)**2
        v_loss = self.value_loss_coeff * valid_mean(value_error, valid)
        v_loss.backward()
        self.v_optimizer.step()
        return v_loss

    def _process_batch(self, agent_inputs, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        return dist_info, value

    def _compute_loss_with_adv(self, loss_inputs, init_rnn_state=None, return_opt_info=False):
        agent_inputs, action, return_, advantage, valid, old_dist_info = loss_inputs
        dist_info, _ = self._process_batch(agent_inputs, init_rnn_state)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info, new_dist_info=dist_info)
        surrogate = ratio * advantage
        pi_loss = -valid_mean(surrogate, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = -self.entropy_loss_coeff * entropy

        loss = pi_loss + entropy_loss

        if return_opt_info:
            perplexity = dist.mean_perplexity(dist_info, valid)
            return loss, entropy, perplexity
        else:
            return loss

    def _compute_kl_constraint(self, loss_inputs, init_rnn_state=None):
        agent_inputs, _, _, _, valid, old_dist_info = loss_inputs
        dist_info, _ = self._process_batch(agent_inputs, init_rnn_state)
        dist = self.agent.distribution

        kl = dist.kl(old_dist_info=old_dist_info, new_dist_info=dist_info)
        kl_constraint = valid_mean(kl, valid)

        return kl_constraint
