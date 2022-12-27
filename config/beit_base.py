_base_ = [
    './_base_/models/upernet_beit.py', './_base_/datasets/recycle.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_160k.py'
]

model = dict(
    pretrained='https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth',
    backbone=dict(
        img_size=(512,512)
    ),
    decode_head=dict(
        num_classes=11
    ),
    auxiliary_head=dict(num_classes=11),
    test_cfg=dict(mode='whole'))


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
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

custom_imports = dict(imports=['sweepLoggerHook'], allow_failed_imports=False)

gpu_ids=[0]