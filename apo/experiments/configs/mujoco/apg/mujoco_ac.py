config = dict(
    agent=dict(),
    algo=dict(
        learning_rate=3e-4,
        lr_eta=0.1,
        rm_vbias_coeff=1.0,
        clip_grad_norm=10.,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        gae_lambda=0.95,
        normalize_advantage=False,
    ),
    env=dict(id="Hopper-v3"),
    model=dict(normalize_observation=False),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=2000 * 10,
    ),
    sampler=dict(
        batch_T=10,
        batch_B=10,
        max_decorrelation_steps=500,
        eval_n_envs=10,
        eval_max_steps=int(1e4),
        eval_max_trajectories=10,
    ),
)
