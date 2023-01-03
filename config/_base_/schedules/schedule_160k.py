# optimizer
optimizer = dict(type='Adam', lr=0.01, betas=(0.9, 0.999), weight_decay=0.0005)
optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
# runtime settings
# runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
evaluation = dict(interval=1, metric='mIoU', pre_eval=True, save_best='mIoU')
