_base_ = [
    './_base_/models/upernet_convnext.py', './_base_/datasets/recycle.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa
model = dict(
    backbone=dict(
        type='mmcls.ConvNeXt',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.3,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=11,
    ),
    auxiliary_head=dict(in_channels=384, num_classes=11),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)),
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    periods=[15, 10, 10, 10, 10],
    restart_weights=[1, 0.85, 0.85, 0.7, 0.7],
    by_epoch=True,
    warmup_by_epoch=True,
    min_lr=5e-6
    )

#Wandb Config
log_config=dict(
    interval=50,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='SweepLoggerHook')
    ]
)

runner = dict(type='EpochBasedRunner', max_epochs=50)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=16)

custom_imports = dict(imports=['sweepLoggerHook','mmcls.models'], allow_failed_imports=False)

gpu_ids=[0]