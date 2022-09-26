_base_ = [
    '../centerpoint/centerpoint_01voxel_second_secfpn_4x8_cyclic_20e_nus.py'
]
use_chamfer, use_num_points, use_fake_voxels = True, True, True
relative_error = False
masking_ratio = 0.7
fake_voxels_ratio = 0.1
loss_weights = dict(
    loss_occupied=1.0,
    loss_num_points_masked=1., #new voxel size, new weight -> (0.5^2)/(0.8^2) 
    loss_chamfer_src_masked=1.,
    loss_chamfer_dst_masked=1.,
    loss_num_points_unmasked=0.,
    loss_chamfer_src_unmasked=0.,
    loss_chamfer_dst_unmasked=0.
)
max_num_gt_points = 100

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]
window_shape = (10, 10, 1) # 12 * 0.32m
drop_info_training = {
    0: {'max_tokens': 10, 'drop_range': (0, 10)},
    1: {'max_tokens': 20, 'drop_range': (10, 20)},
    2: {'max_tokens': 30, 'drop_range': (20, 30)},
    3: {'max_tokens': 50, 'drop_range': (30, 50)},
    4: {'max_tokens': 100, 'drop_range': (50, 100000)},
}
drop_info_test = {
    0: {'max_tokens': 30, 'drop_range': (0, 30)},
    1: {'max_tokens': 60, 'drop_range': (30, 60)},
    2: {'max_tokens': 100, 'drop_range': (60, 100)},
    3: {'max_tokens': 200, 'drop_range': (100, 200)},
    4: {'max_tokens': 256, 'drop_range': (200, 100000)},
}
drop_info = (drop_info_training, drop_info_test)

model = dict(
    pts_voxel_layer=dict(point_cloud_range=point_cloud_range, voxel_size=voxel_size),
    pts_voxel_encoder=dict(
    ),
    pts_middle_encoder=dict(
        type='SparseEncoderMasked',
        in_channels=5,
        sparse_shape=[41, 1024, 1024],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
        window_shape=window_shape,
        sparse_shape_sst=(128, 128, 1),
        shuffle_voxels=True,
        debug=False,
        drop_info=drop_info,
        pos_temperature=10000,
        normalize_pos=False,
        mute=True,
        masking_ratio=masking_ratio,
        drop_points_th=max_num_gt_points,
        pred_dims=3,  # x, y, z
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        fake_voxels_ratio=fake_voxels_ratio,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),

    decoder=dict(
        type='SSTv2Decoder',
        d_model=[128, ] * 2,
        nhead=[8, ] * 2,
        num_blocks = 2,
        dim_feedforward=[256, ] * 2,
        output_shape=[128, 128],
        debug=False,
        use_fake_voxels=use_fake_voxels,
        in_channel=512,
        normalize_input=True,
        #layer_cfg=dict(
        #    post_norm=False,
        #)
    ),

    pts_bbox_head=dict(
        _delete_=True,
        type='ReconstructionHead',
        in_channels=128,
        feat_channels=128,
        num_chamfer_points=26, #new voxel size -> (0.8^2)/(0.5^2)
        pred_dims=3,
        only_masked=True,
        relative_error=relative_error,
        loss_weights=loss_weights,
        use_chamfer=use_chamfer,
        use_num_points=use_num_points,
        use_fake_voxels=use_fake_voxels,
        max_num_gt_points=max_num_gt_points,
    )
)

# runtime settings
epochs = 20
runner = dict(type='EpochBasedRunner', max_epochs=epochs)
evaluation = dict(interval=epochs+1)  # Don't evaluate when doing pretraining
workflow = [("train", 1), ("val", 1)]  # But calculate val loss after each epoch
checkpoint_config = dict(interval=2)

fp16 = dict(loss_scale=32.0)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.01)
lr_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(10, 1e-4),
    cyclic_times=1,
    step_ratio_up=0.4,
)
momentum_config = dict(
    _delete_=True,
    policy='cyclic',
    target_ratio=(0.85 / 0.95, 1),
    cyclic_times=1,
    step_ratio_up=0.4,
)