"""
Runs one instance of the environment and optimizes using the PPO algorithm. 
Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.

Requires OpenAI gym (and maybe mujoco).  If not installed, move on to next
example.

"""
import numpy as np

from apo.experiments.configs.mujoco.pg.mujoco_ppo import config
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.algos.pg.ppo import PPO
from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging.context import logger_context


def build_and_train(env_id="Swimmer-v3", run_ID=0, cuda_idx=None):
    config["env"]["id"] = env_id
    config["algo"]["normalize_advantage"] = True
    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        **config["sampler"],
    )
    algo = PPO(optim_kwargs=config["optim"], **config["algo"])
    agent = MujocoFfAgent(model_kwargs=config["model"], **config["agent"])
    n_cpus = config["sampler"]["batch_B"]
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=np.random.choice(np.arange(128), n_cpus).tolist()),
        seed=run_ID,
        **config["runner"],
    )
    name = "pponorm_" + env_id
    log_dir = f"data/ppo_norm/{env_id}"
    with logger_context(
            log_dir,
            run_ID,
            name,
            config,
            override_prefix=True,
            use_summary_writer=True,
    ):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='Swimmer-v3')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    args = parser.parse_args()
    build_and_train(
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
    )
