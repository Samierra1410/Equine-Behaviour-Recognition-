dataset_type = 'AP10KDataset'
data_mode = 'topdown'
data_root = 'data/ap10k/'
codec = dict(type='MSRAHeatmap', input_size=(256, 256), heatmap_size=(64, 64), sigma=2)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(type='PoseDataPreprocessor', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(type='HRNet', in_channels=3, extra=dict(
        stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,), num_channels=(64,)),
        stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4,4), num_channels=(32,64)),
        stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4,4,4), num_channels=(32,64,128)),
        stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4,4,4,4), num_channels=(32,64,128,256)))),
    head=dict(type='HeatmapHead', in_channels=32, out_channels=17, deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True), decoder=codec),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))
test_dataloader = dict(batch_size=1, num_workers=1, persistent_workers=False, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type, data_root=data_root, data_mode=data_mode,
        ann_file='annotations/ap10k-val-split1.json', data_prefix=dict(img='data/'),
        test_mode=True, pipeline=[dict(type='LoadImage'), dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=codec['input_size']), dict(type='PackPoseInputs')]))
test_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/ap10k-val-split1.json')
