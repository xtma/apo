import torch

from rlpyt.utils.misc import zeros


def discount_return(reward, done, bootstrap_value, eta, discount, return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc. Computes
    discounted sum of future rewards from each time-step to the end of the
    batch, including bootstrapping value.  Sum resets where `done` is 1.
    Optionally, writes to buffer `return_dest`, if provided.  Operations
    vectorized across all trailing dimensions after the first [T,]."""
    return_ = return_dest if return_dest is not None else zeros(reward.shape, dtype=reward.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    return_[-1] = reward[-1] - eta + discount * bootstrap_value * nd[-1]
    for t in reversed(range(len(reward) - 1)):
        return_[t] = reward[t] - eta + discount * return_[t + 1] * nd[t]
    return return_


def generalized_advantage_estimation(reward,
                                     value,
                                     done,
                                     bootstrap_value,
                                     eta,
                                     discount,
                                     gae_lambda,
                                     advantage_dest=None,
                                     return_dest=None):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""
    advantage = advantage_dest if advantage_dest is not None else zeros(reward.shape, dtype=reward.dtype)
    return_ = return_dest if return_dest is not None else zeros(reward.shape, dtype=reward.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    advantage[-1] = reward[-1] - eta + discount * bootstrap_value * nd[-1] - value[-1]
    for t in reversed(range(len(reward) - 1)):
        delta = reward[t] - eta + discount * value[t + 1] * nd[t] - value[t]
        advantage[t] = delta + discount * gae_lambda * nd[t] * advantage[t + 1]
    return_[:] = advantage + value
    return advantage, return_
