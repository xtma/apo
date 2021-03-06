config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=3e-4,
        clip_grad_norm=10.,
        entropy_loss_coeff=0.01,
        gae_lambda=0.95,
        minibatches=20,
        epochs=10,
        ratio_clip=0.2,
        normalize_advantage=False,
        linear_lr_schedule=False,
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
