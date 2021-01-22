"""
Launches multiple experiment runs and organizes them on the local
compute resource.
Processor (CPU and GPU) affinities are all specified, to keep each
experiment on its own hardware without interference.  Can queue up more
experiments than fit on the machine, and they will run in order over time.  

To understand rules and settings for affinities, try using 
affinity = affinity.make_affinity(..)
OR
code = affinity.encode_affinity(..)
slot_code = affinity.prepend_run_slot(code, slot)
affinity = affinity.affinity_from_code(slot_code)
with many different inputs to encode, and see what comes out.

The results will be logged with a folder structure according to the
variant levels constructed here.

"""

from itertools import product

from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import VariantLevel, make_variants

# Either manually set the resources for the experiment:
# affinity_code = encode_affinity(
#     n_cpu_core=128,
#     n_gpu=8,
#     contexts_per_gpu=2,
#     # hyperthread_offset=8,  # if auto-detect doesn't work, number of CPU cores
#     # n_socket=1,  # if auto-detect doesn't work, can force (or force to 1)
#     # cpu_per_run=1,
#     set_affinity=True,  # it can help to restrict workers to individual CPUs
# )
# Or try an automatic one, but results may vary:
affinity_code = quick_affinity_code(
    n_parallel=None,
    use_gpu=True,
    contexts_per_gpu=8,
)
runs_per_setting = 5
variant_levels = list()

# Within a variant level, list each combination explicitly.

# ------------------APPO------------------ #
# APPO
experiment_title = "appo_0107"
# discount = [0.9, 0.95, 0.99, 0.999, 1.0]
discount = [1.0]
gae_lambda = [0.8, 0.9, 0.95, 0.99]
lr_eta = [0.03, 0.1, 0.3]
rm_vb_coef = [0., 0.03, 0.1, 0.3, 1.0]
env_id = ['Swimmer-v3', 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Humanoid-v3']

values = list(product(discount, gae_lambda, lr_eta, rm_vb_coef, env_id))
dir_names = ["data/appo_g-{}_l-{}_e-{}_v-{}_{}".format(*v) for v in values]
keys = [
    ("algo", "discount"),
    ("algo", "gae_lambda"),
    ("algo", "lr_eta"),
    ("algo", "rm_vbias_coeff"),
    ("env", "id"),
]
variant_levels.append(VariantLevel(keys, values, dir_names))
# ---------------------------------------- #

# ------------------PPO------------------- #
# experiment_title = "ppo_0107"
# discount = [0.9, 0.95, 0.99, 0.999]
# gae_lambda = [0.8, 0.9, 0.95, 0.99]
# env_id = ['Swimmer-v3', 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Humanoid-v3']
# longrun = [False]

# values = list(product(discount, gae_lambda, env_id, longrun))
# dir_names = ["data/ppo_g-{}_l-{}_{}".format(*v) for v in values]

# keys = [
#     ("algo", "discount"),
#     ("algo", "gae_lambda"),
#     ("env", "id"),
#     ("algo", "longrun"),
# ]
# variant_levels.append(VariantLevel(keys, values, dir_names))
# ---------------------------------------- #

# Between variant levels, make all combinations.
variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script="exp.py",
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    override_prefix=True,
)
