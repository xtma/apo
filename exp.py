"""
Runs one instance of the environment and optimizes using the specific algorithm.
Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.
"""
from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRlEval
# from rlpyt.samplers.parallel.gpu.sampler import GpuSampler as Sampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler as Sampler
# from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler as Sampler
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.logging.context import logger_context

# from rlpyt.agents.pg.mujoco import MujocoFfAgent as Agent
from apo.agents.mujoco_eval import MujocoFfEvalAgent as Agent
from apo.algos.apg.appo import APPO as Algo
from apo.envs.traj_info import AverageTrajInfo
from apo.experiments.configs.mujoco.apg.mujoco_appo import config


def build_and_train(slot_affinity_code, log_dir, run_ID):

    affinity = affinity_from_code(slot_affinity_code)
    variant = load_variant(log_dir)
    global config
    config = update_config(config, variant)

    sampler = Sampler(
        EnvCls=env_cls,
        env_kwargs=config['env'],
        eval_env_kwargs=config['env'],
        TrajInfoCls=AverageTrajInfo,
        **config["sampler"],
    )
    algo = Algo(optim_kwargs=config["optim"], **config["algo"])
    agent = Agent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        seed=int(run_ID),
        **config["runner"],
    )
    with logger_context(
            log_dir,
            run_ID,
            log_dir.split('/')[-1],
            config,
            override_prefix=True,
            use_summary_writer=True,
    ):
        runner.train()


if __name__ == "__main__":
    import sys
    build_and_train(*sys.argv[1:])
