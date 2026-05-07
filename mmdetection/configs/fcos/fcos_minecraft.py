_base_ = 'fcos_r50-caffe_fpn_gn-head_1x_coco.py'

classes = ('bee','chicken','cow','creeper','enderman','fox','frog','ghast',
           'goat','llama','pig','sheep','skeleton','spider','turtle','wolf','zombie')

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://detectron2/resnet50_caffe')
    ),
    bbox_head=dict(num_classes=len(classes)),
)

data_root = 'datasets/minecraft/'


metainfo = dict(classes=classes)

# Даталодеры
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/annotations.json',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='valid/annotations.json',
        data_prefix=dict(img='valid/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='PackDetInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader

# Evaluator
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/annotations.json',
    metric='bbox',
    format_only=False,
)

test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.333, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, milestones=[8, 11], gamma=0.1),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer',
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP', rule='greater')
)

work_dir = '/content/drive/MyDrive/minecraft_project/artifacts/fcos'