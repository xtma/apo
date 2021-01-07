"""
Runs one instance of the environment and optimizes using the specific algorithm.
Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.
"""
import numpy as np

from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.envs.gym import make as gym_make
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.utils.logging.context import logger_context
from apo.envs.traj_info import AverageTrajInfo


def build_and_train(
    algo_name='appo',
    env_id="Swimmer-v3",
    run_ID=0,
    cuda_idx=None,
    gamma=0.9,
    lamda=0.8,
    lr_eta=0.1,
    rm_vb_coef=0.1,
):

    if algo_name == "appo":
        from apo.algos.apg.appo import APPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
    elif algo_name == "appo2":
        from apo.algos.apg.appo2 import APPO2 as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
    elif algo_name == "aac":
        from apo.algos.apg.aac import AAC as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_aac import config
    elif algo_name == "atrpo":
        from apo.algos.apg.atrpo import ATRPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_atrpo import config
    elif algo_name == "ppo":
        from apo.algos.apg.appo import APPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
        config["algo"]["longrun"] = False
    elif algo_name == "ppo_norm":
        from apo.algos.apg.appo import APPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
        config["algo"]["longrun"] = False
        config["algo"]["normalize_advantage"] = True
    elif algo_name == "trpo":
        from apo.algos.apg.atrpo import ATRPO as Algo
        from apo.experiments.configs.mujoco.apg.mujoco_appo import config
        config["algo"]["longrun"] = False
    else:
        assert NotImplementedError

    config["env"]["id"] = env_id
    config["algo"]["discount"] = gamma
    config["algo"]["gae_lambda"] = lamda
    config["algo"]["lr_eta"] = lr_eta
    config["algo"]["rm_vbias_coeff"] = rm_vb_coef

    sampler = GpuSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        TrajInfoCls=AverageTrajInfo,
        **config["sampler"],
    )
    algo = Algo(optim_kwargs=config["optim"], **config["algo"])
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
    if algo_name in ['ppo', 'trpo']:
        name = f"{algo_name}_g-{gamma}_l-{lamda}_{env_id}"
    else:
        name = f"{algo_name}_g-{gamma}_l-{lamda}_e-{lr_eta}_v-{rm_vb_coef}_{env_id}"
    log_dir = f"data1/{name}"
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
    parser.add_argument('--algo', help='algorithm', default='ppo')
    parser.add_argument('--env_id', help='environment ID', default='Swimmer-v3')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--gamma', help='discount', type=float, default=0.99)
    parser.add_argument('--lamda', help='gae lambda', type=float, default=0.95)
    parser.add_argument('--lr_eta', help='lr_eta', type=float, default=0.1)
    parser.add_argument('--rm_vb_coef', type=float, default=0.1)
    args = parser.parse_args()
    build_and_train(
        algo_name=args.algo,
        env_id=args.env_id,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        gamma=args.gamma,
        lamda=args.lamda,
        lr_eta=args.lr_eta,
        rm_vb_coef=args.rm_vb_coef,
    )
