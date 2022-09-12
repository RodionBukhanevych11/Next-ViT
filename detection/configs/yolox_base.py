
img_scale = (736, 736)  # height, width

# model settings
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='nextvit_base',
                  norm_eval=True,
                  with_extra_norm=False,),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=3, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/media/data2/vv/dvc_datasets/dataset_ppe/for_experiments/seah_hardhat_coco/'
classes = ('in_hardhat','not_in_hardhat','hardhat_unrecognized')

train_pipeline = [
    #dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    #dict(
    #    type='Pad',
    #    pad_to_square=True,
    #    # If the image is three-channel, the pad value needs
    #    # to be set separately for each channel.
    #    pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        ann_file=data_root+'ann/hardhat/train.json',
        img_prefix=data_root+'images/',
        classes = classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Pad',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type='CocoDataset',
        ann_file=data_root+'ann/hardhat/val.json',
        img_prefix=data_root+'images/',
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        ann_file=data_root+'ann/hardhat/test.json',
        img_prefix=data_root+'images/',
        classes = classes,
        pipeline=test_pipeline))

# optimizer
# default 8 gpu
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            bn=dict(decay_mult=0.0), norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)

max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10
workflow = [('train', 1), ('val', 1)]

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.000001)

runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

'''
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        ),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        ),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        )
]
'''

evaluation = dict(metric=['bbox'])
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
find_unused_parameters=True

checkpoint_config = dict(interval=interval)
evaluation = dict(
    save_best='auto',
    # The evaluation interval is 'interval' when running epoch is
    # less than ‘max_epochs - num_last_epochs’.
    # The evaluation interval is 1 when running epoch is greater than
    # or equal to ‘max_epochs - num_last_epochs’.
    interval=interval,
    dynamic_intervals=[(max_epochs - num_last_epochs, 1)],
    metric='bbox')
#log_config = dict(interval=50)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=12)
gpu_ids = range(0,2)