lr=1e-5 #1e-4
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.9, 0.999),  # the momentum is change during training
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}),
    )
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 1e-3), #100, 1e-3
    cyclic_times=1,
    step_ratio_up=0.1,
)
momentum_config = None # dict(
    #policy='cyclic',
    #target_ratio=(0.8947368421052632, 1),
    #cyclic_times=1,
    #step_ratio_up=0.1)
runner = dict(type='EpochBasedRunner', max_epochs=24)