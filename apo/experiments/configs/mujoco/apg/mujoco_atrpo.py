config = dict(
    agent=dict(),
    algo=dict(
        longrun=True,
        lr_eta=0.1,
        rm_vbias_coeff=1.0,
        max_constraint_value=0.01,
        entropy_loss_coeff=0.01,
        gae_lambda=0.95,
        minibatches=20,
        epochs=10,
        normalize_advantage=False,
        bootstrap_timelimit=True,
    ),
    env=dict(id="Hopper-v3"),
    model=dict(normalize_observation=False),
    optim=dict(),
    runner=dict(
        n_steps=3e6,
        log_interval_steps=2000 * 10,
    ),
    sampler=dict(
        batch_T=200,
        batch_B=10,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(1e4),
        eval_max_trajectories=10,
    ),
)