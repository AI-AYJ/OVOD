_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'], allow_failed_imports=False)

# hyper-parameters
num_classes = 20
num_training_classes = 20
max_epochs = 200  # Maximum training epochs
close_mosaic_epochs = 190
save_epoch_intervals = 10
text_channels = 512
neck_embed_channels = [128, 128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 4, 8, _base_.last_stage_out_channels // 2 // 32]
base_lr = 1e-3
weight_decay = 0.0005
train_batch_size_per_gpu = 16
load_from = 'pretrained_models/yolo_world_v2_l_vlpan_bn_sgd_1e-3_40e_8gpus_finetune_coco_ep80-e1288152.pth'
text_model_name = '../pretrained_models/clip-vit-base-patch32-projection'
text_model_name = 'openai/clip-vit-base-patch32'
persistent_workers = False


wotr_classes = (
    'tree','red_light','green_light','crosswalk','blind_road','sign','person',
    'bicycle','bus','truck','car','motorcycle','reflective_cone','ashcan',
    'warning_column','roadblock','pole','dog','tricycle','fire_hydrant'
)


# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),

    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model=dict(
            **_base_.model.backbone,
            out_indices=(1, 2, 3, 4)
        ),
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name=text_model_name,
            frozen_modules=['all']
        )
    ),

        neck=dict(type='YOLOWorldDualPAFPN',
              guide_channels=text_channels,
              embed_channels=neck_embed_channels,
              num_heads=neck_num_heads,
              in_channels=[128, 256, 512, 512],
              out_channels=[128, 256, 512, 512],
              block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
              text_enhancder=dict(type='ImagePoolingAttentionModule',
                                  embed_channels=256,
                                  num_heads=8)),

    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes,
            in_channels=[128, 256, 512, 512],
            featmap_strides=[4, 8, 16, 32],
            p2_cls_weight=1.0,
            p2_bbox_weight=1.0,
        ),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator',
            offset=0.5,
            strides=[4, 8, 16, 32],
        ),
    ),

    train_cfg=dict(
        assigner=dict(
            num_classes=num_training_classes
        )
)
)
# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
mosaic_affine_transform = [
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *_base_.pre_transform, *mosaic_affine_transform,
    dict(type='YOLOv5MultiModalMixUp',
         prob=_base_.mixup_prob,
         pre_transform=[*_base_.pre_transform, *mosaic_affine_transform]),
    *_base_.last_transform[:-1], *text_transform
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]


# --- remove mmdet.Albu to avoid albumentations dependency/key conflicts ---
def _drop_albu(pipeline):
    return [
        t for t in pipeline
        if not (isinstance(t, dict) and t.get('type') in ('mmdet.Albu', 'Albu'))
    ]

train_pipeline = _drop_albu(train_pipeline)
train_pipeline_stage2 = _drop_albu(train_pipeline_stage2)


wotr_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/heat/young/WOTR_YOLO/',
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=wotr_classes),
    ),
    class_text_path='/home/heat/young/WOTR_YOLO/texts/wotr_class_texts.json',
    pipeline=train_pipeline,
)


train_dataloader = dict(
    persistent_workers=persistent_workers,
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=wotr_train_dataset,
)

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

wotr_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/heat/young/WOTR_YOLO/',
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=dict(classes=wotr_classes),
    ),
    class_text_path='/home/heat/young/WOTR_YOLO/texts/wotr_class_texts.json',
    pipeline=test_pipeline,
)


val_dataloader = dict(dataset=wotr_val_dataset)
wotr_test_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/home/heat/young/WOTR_YOLO/',
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        metainfo=dict(classes=wotr_classes),
    ),
    class_text_path='/home/heat/young/WOTR_YOLO/texts/wotr_class_texts.json',
    pipeline=test_pipeline,
)

test_dataloader = dict(dataset=wotr_test_dataset)

# training settings
default_hooks = dict(param_scheduler=dict(scheduler_type='linear',
                                          lr_factor=0.01,
                                          max_epochs=max_epochs),
                     checkpoint=dict(max_keep_ckpts=5,
                                     save_best='coco/bbox_mAP',
                                     interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]

train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])

optim_wrapper = dict(optimizer=dict(
    _delete_=True,
    type='SGD',
    lr=base_lr,
    momentum=0.937,
    nesterov=True,
    weight_decay=weight_decay,
    batch_size_per_gpu=train_batch_size_per_gpu),
                     paramwise_cfg=dict(
                         custom_keys={
                             'backbone.text_model': dict(lr_mult=0.01),
                             'logit_scale': dict(weight_decay=0.0)
                         }),
                     constructor='YOLOWv5OptimizerConstructor')

# evaluation settings
val_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/home/heat/young/WOTR_YOLO/annotations/instances_val.json',
    metric='bbox'
)

test_evaluator = dict(
    _delete_=True,
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file='/home/heat/young/WOTR_YOLO/annotations/instances_test.json',
    metric='bbox'
)
