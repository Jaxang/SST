import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn as nn

from mmdet3d.models.middle_encoders.sst_input_layer_v2 import SSTInputLayerV2
from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
from mmdet3d.ops import spconv as spconv
from mmdet3d.ops.spconv import SparseModule
from ..builder import MIDDLE_ENCODERS
from mmdet3d.ops import flat2window_v2, window2flat_v2, get_inner_win_inds


@MIDDLE_ENCODERS.register_module()
class SparseEncoderMasked(SSTInputLayerV2):
    def __init__(
        self,
        in_channels=5,
        sparse_shape=(41, 1024, 1024),
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
        sparse_shape_sst=(128, 128, 1),
        masking_ratio=0.7,
        drop_points_th=100,
        pred_dims=3,
        fake_voxels_ratio=0.1,
        use_chamfer=True,
        use_num_points=True,
        use_fake_voxels=True,
        voxel_size=(0.2, 0.2, 4),
        voxel_size_dec=(0.8, 0.8, 8),
        point_cloud_range=(0, -40, -3, 70.4, 40, 1),
        shuffle_voxels=True,
        debug=True,
        normalize_pos=False,
        pos_temperature=10000,
        mute=False,
        drop_info=None,
        window_shape=None,
        masked_window_shape=None,
    ):
        super().__init__(
            drop_info,
            window_shape,
            sparse_shape_sst,
            shuffle_voxels=shuffle_voxels,
            debug=debug,
            normalize_pos=normalize_pos,
            pos_temperature=pos_temperature,
            mute=mute,
        )
        self.masking_ratio = masking_ratio
        self.drop_points_th = drop_points_th
        self.pred_dims = pred_dims
        self.voxel_size = voxel_size_dec
        self.fake_voxels_ratio = fake_voxels_ratio
        self.use_chamfer = use_chamfer
        self.use_num_points = use_num_points
        self.use_fake_voxels = use_fake_voxels
        self.masked_window_shape = masked_window_shape
        self.unmasked_window_shape = window_shape
        assert (
            use_chamfer or use_num_points or use_fake_voxels
        ), "Need to use at least one of chamfer, num_points, and fake_voxels"

        self.vx = voxel_size_dec[0]
        self.vy = voxel_size_dec[1]
        self.vz = voxel_size_dec[2]
        self.voxel_size_dec = voxel_size_dec
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        assert (
            use_chamfer or use_num_points or use_fake_voxels
        ), "Need to use at least one of chamfer, num_points, and fake_voxels"

        # Sparse encoder init
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape_enc = sparse_shape
        self.sparse_shape = sparse_shape_sst
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

        self.masked = True
        self.proj = nn.Linear(in_channels, self.output_channels)

    @force_fp32(apply_to=("voxel_features"), out_fp16=False)
    def forward(self, voxel_features, coors, batch_size, voxels, num_points):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.
            voxels (torch.float32): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.int32): Number of points in each voxel in shape (N,).

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        gt_dict, fake_voxel_coors = self.get_ground_truth(
            voxel_features, coors, batch_size, voxels, num_points
        )
        voxel_info_decoder, voxel_info_encoder = self.mask_voxels(
            voxel_features.device, coors, voxel_features, fake_voxel_coors, gt_dict
        )
        voxel_info_decoder["gt_dict"] = gt_dict
        voxel_info_encoder["voxel_info_decoder"] = voxel_info_decoder

        input_sp_tensor = spconv.SparseConvTensor(
            voxel_info_encoder["voxel_feats"],
            voxel_info_encoder["coors"],
            self.sparse_shape_enc,
            batch_size,
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features.half(), voxel_info_encoder

    def get_ground_truth(self, voxel_features, coors, batch_size, voxels, num_points):
        """
        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.
            voxels (torch.float32): Point features in shape
                (N, M, 3(4)). N is the number of voxels and M is the maximum
                number of points inside a single voxel.
            num_points (torch.int32): Number of points in each voxel in shape (N,).

        Returns:
            dict: Backbone features.
        """
        gt_dict = {}
        vx, vy, vz = self.sparse_shape  # TODO: check this
        max_num_voxels = batch_size * vx * vy * vz

        # [b,z,y,x] to idx in maximum sized flat tensor
        voxel_indices = self.get_voxel_indices(coors)
        point_coors = coors.unsqueeze(1).repeat(1, voxels.shape[1], 1)
        point_indices = voxel_indices.unsqueeze(1).expand(
            -1, voxels.shape[1]
        )  # (N,) -> (N, M)

        non_padded_point_mask = voxels.abs().sum(dim=-1) != 0
        non_padded_points = voxels[non_padded_point_mask]
        non_padded_point_coors = point_coors[non_padded_point_mask]
        non_padded_point_indices = point_indices[non_padded_point_mask]

        # Reassign coors from small voxels to large voxels
        div = [
            enc / dec
            for enc, dec in zip(self.sparse_shape_enc, reversed(self.sparse_shape))
        ]
        big_voxel_coors = non_padded_point_coors // torch.tensor(
            [1] + div, device=non_padded_point_coors.device
        )
        big_voxel_coors = big_voxel_coors.int()

        big_voxel_idx = self.get_voxel_indices(big_voxel_coors)
        shuffle = torch.argsort(
            torch.rand(len(big_voxel_idx))
        )  # Shuffle to drop random points
        restore = torch.argsort(shuffle)
        # Assign a unique id to each point within a voxel
        inner_coors = get_inner_win_inds(big_voxel_idx[shuffle])[restore]

        # Only keep up to self.drop_points_th points per voxel
        keep = inner_coors < self.drop_points_th
        non_padded_points = non_padded_points[keep]
        non_padded_point_coors = non_padded_point_coors[keep]
        big_voxel_idx = big_voxel_idx[keep]
        inner_coors = inner_coors[keep]
        big_voxel_coors = big_voxel_coors[keep]

        big_voxels = torch.zeros(
            (max_num_voxels, self.drop_points_th, voxels.shape[2]), device=voxels.device
        )
        big_voxels[big_voxel_idx, inner_coors] = non_padded_points

        points_per_voxel = (big_voxels.abs().sum(dim=-1) != 0).sum(dim=-1)
        idx_non_empty_voxels = points_per_voxel.nonzero().ravel()
        points_per_voxel = points_per_voxel[idx_non_empty_voxels]
        big_voxels = big_voxels[idx_non_empty_voxels]
        big_voxel_coors = self.get_voxel_coors(idx_non_empty_voxels)

        # Points per voxel
        if self.use_num_points:
            gt_dict["num_points_per_voxel"] = points_per_voxel
            assert (points_per_voxel > 0).all(), "Exists voxel without connected points"

        # Get points per voxel
        if self.use_chamfer:
            # Calculate points position relative to voxel center
            point_rel_voxel_center = torch.zeros_like(big_voxels[..., :3])
            point_rel_voxel_center[:, :, 0] = big_voxels[:, :, 0] - (
                big_voxel_coors[:, 3].type_as(big_voxels).unsqueeze(1) * self.vx
                + self.x_offset
            )
            point_rel_voxel_center[:, :, 1] = big_voxels[:, :, 1] - (
                big_voxel_coors[:, 2].type_as(big_voxels).unsqueeze(1) * self.vy
                + self.y_offset
            )
            point_rel_voxel_center[:, :, 2] = big_voxels[:, :, 2] - (
                big_voxel_coors[:, 1].type_as(big_voxels).unsqueeze(1) * self.vz
                + self.z_offset
            )

            points_rel_norm = 2 / torch.tensor(
                self.voxel_size, device=big_voxels.device
            ).view(1, 1, -1)
            point_rel_voxel_center = (
                point_rel_voxel_center * points_rel_norm
            )  # x,y,z in [-1,1]

            gt_dict["points_per_voxel"] = point_rel_voxel_center
            gt_dict["points_per_voxel_padding"] = (
                big_voxels.abs().sum(dim=-1) == 0
            )  # 1=padded point, 0=real point
            gt_dict["voxel_coors"] = big_voxel_coors
            gt_dict["gt_points"] = big_voxels[
                ~gt_dict["points_per_voxel_padding"]
            ]  # For visualization
            gt_dict["gt_point_coors"] = big_voxel_coors.unsqueeze(1).repeat(
                1, self.drop_points_th, 1
            )[
                ~gt_dict["points_per_voxel_padding"]
            ]  # For visualization

        # TODO: Potentially add fake voxels
        fake_voxel_coors = None
        if self.use_fake_voxels:
            max_num_voxels_per_batch = vx * vy * vz
            voxels_per_batch = torch.bincount(
                big_voxel_coors[:, 0].long()
            )  # e.g [5000, 6020, 4920, 5107] for batch_size=4
            n_fake_voxels_per_batch = (
                voxels_per_batch * self.fake_voxels_ratio
            ).long()  # e.g [500, 602, 492, 510] for fake_voxel_ratio=0.1

            empty_voxels = torch.ones(
                max_num_voxels, device=voxels.device, dtype=torch.bool
            )
            empty_voxels[idx_non_empty_voxels] = False
            empty_voxels = empty_voxels.view(batch_size, -1)

            fake_voxel_idxs = [
                torch.where(empty_voxels[b])[0] + (b * max_num_voxels_per_batch)
                for b in range(batch_size)
            ]
            fake_voxel_idxs = torch.cat(
                [
                    idx[torch.randperm(len(idx), device=big_voxels.device)][
                        :n_fake
                    ]  # Shuffle and take n_fake first indices
                    for i, (idx, n_fake) in enumerate(
                        zip(fake_voxel_idxs, n_fake_voxels_per_batch)
                    )
                ]
            )
            fake_voxel_coors = self.get_voxel_coors(fake_voxel_idxs)

            mask = torch.zeros(
                (len(big_voxel_coors) + len(fake_voxel_coors)),
                device=big_voxel_coors.device,
                dtype=torch.bool,
            )
            mask[len(big_voxel_coors) :] = True
            gt_dict["fake_voxel_mask"] = mask

        return gt_dict, fake_voxel_coors

    def mask_voxels(
        self, device, voxel_coors, voxel_features, fake_voxel_coors, gt_dict
    ):
        """
        Randomly mask voxels

        Args:
            device: device
            voxel_coors: shape=[N, 4], [b, z, y, x], voxel coordinate for each voxel
            voxel_features: shape=[N, C], N is the voxel num in the batch.
            fake_voxel_coors: shape=[N, 4], [b, z, y, x], voxel coordinate for fake voxels
        """

        # Masking voxels: True -> masked, False -> unmasked
        big_voxel_features = gt_dict["points_per_voxel"].sum(dim=1)/gt_dict["num_points_per_voxel"].unsqueeze(-1)
        missing_features = torch.zeros((len(big_voxel_features), voxel_features.shape[-1]-big_voxel_features.shape[-1]), device=device)
        big_voxel_features = torch.cat((big_voxel_features, missing_features), dim=-1)
        mask = torch.rand(len(big_voxel_features), device=device) < self.masking_ratio
        big_voxel_coors = gt_dict["voxel_coors"]

        # Add fake voxels
        if self.use_fake_voxels:
            fake_voxel_idx = torch.arange(len(fake_voxel_coors), device=device) + len(
                big_voxel_coors
            )
            big_voxel_coors = torch.cat([big_voxel_coors, fake_voxel_coors], dim=0)
            fake_voxel_feats = torch.zeros(
                (len(fake_voxel_coors), big_voxel_features.shape[1]),
                device=device,
                dtype=voxel_features.dtype,
            )
            big_voxel_features = torch.cat([big_voxel_features, fake_voxel_feats], dim=0)
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(len(fake_voxel_coors), device=device, dtype=torch.bool), #TODO: maybe change to ones?
                ]
            )
            n_fake_voxels = len(fake_voxel_idx)

        # Get info for decoder input, Might drop voxels
        # We also apply projection to the voxel features for matching the feature dimension
        # of the decoder
        voxel_info_decoder = super().forward(
            self.proj(big_voxel_features), big_voxel_coors, batch_size=None
        )
        assert len(voxel_info_decoder["voxel_feats"]) == len(
            big_voxel_features
        ), "Dropping is not allowed for reconstruction"

        big_masked_idx, big_unmasked_idx = mask.nonzero().ravel(), (~mask).nonzero().ravel()
        big_n_masked_voxels, big_n_unmasked_voxels = len(big_masked_idx), len(big_unmasked_idx)

        div = [
            enc / dec
            for enc, dec in zip(self.sparse_shape_enc, reversed(self.sparse_shape))
        ]
        voxel_coors_in_big_voxels = voxel_coors // torch.tensor(
                    [1] + div, device=voxel_coors.device
                )
        voxel_coors_in_big_voxels = voxel_coors_in_big_voxels.int()
        masked_idx = sum((voxel_coors_in_big_voxels==x).all(dim=-1) for x in big_voxel_coors[big_masked_idx]).nonzero().ravel()
        unmasked_idx = sum((voxel_coors_in_big_voxels==x).all(dim=-1) for x in big_voxel_coors[big_unmasked_idx]).nonzero().ravel()
        n_masked_voxels, n_unmasked_voxels = len(masked_idx), len(unmasked_idx)

        unmasked_voxels = voxel_features[unmasked_idx]
        unmasked_voxel_coors = voxel_coors[unmasked_idx]

        voxel_info_encoder = {
            "voxel_feats": unmasked_voxels,
            "coors": unmasked_voxel_coors,
        }

        if self.masked_window_shape is not None:
            # Change window size for masked encoder
            assert (
                self.unmasked_window_shape is not None
            ), "Cannot restore the window shape"
            self.window_shape = self.masked_window_shape

        if self.masked_window_shape:
            # Change back the window size for the next iteration
            self.window_shape = self.unmasked_window_shape

        voxel_info_decoder["mask"] = mask
        voxel_info_decoder["n_unmasked"] = big_n_unmasked_voxels
        voxel_info_decoder["n_masked"] = big_n_masked_voxels
        voxel_info_decoder["unmasked_idx"] = big_unmasked_idx
        voxel_info_decoder["masked_idx"] = big_masked_idx
        if self.use_fake_voxels:
            voxel_info_decoder["fake_voxel_idx"] = fake_voxel_idx
            voxel_info_decoder["n_fake"] = n_fake_voxels

        # Index mapping from decoder output to other
        dec2dec_input_idx = torch.argsort(voxel_info_decoder["original_index"])
        dec2masked_idx = dec2dec_input_idx[big_masked_idx]
        dec2unmasked_idx = dec2dec_input_idx[big_unmasked_idx]
        dec2enc_idx = dec2unmasked_idx[
            torch.arange(len(big_unmasked_idx), device=unmasked_voxels.device, dtype=torch.long)
        ]
        voxel_info_decoder["dec2input_idx"] = dec2dec_input_idx
        voxel_info_decoder["dec2unmasked_idx"] = dec2unmasked_idx
        voxel_info_decoder["dec2masked_idx"] = dec2masked_idx
        voxel_info_decoder["dec2enc_idx"] = dec2enc_idx
        if self.use_fake_voxels:
            dec2fake_idx = dec2dec_input_idx[fake_voxel_idx]
            voxel_info_decoder["dec2fake_idx"] = dec2fake_idx

        # Debug - sanity check
        if self.debug:
            decoder_feats = voxel_info_decoder["voxel_feats"]
            decoder_coors = voxel_info_decoder["voxel_coors"]

            assert torch.allclose(
                decoder_feats[dec2dec_input_idx], voxel_features
            ), "The mapping from decoder to decoder input is invalid"
            assert torch.allclose(
                decoder_coors[dec2dec_input_idx], voxel_coors.long()
            ), "The mapping from decoder to decoder input is invalid"

            assert torch.allclose(
                decoder_feats[dec2masked_idx], voxel_features[masked_idx]
            ), "The mapping from decoder to masked input is invalid"
            assert torch.allclose(
                decoder_coors[dec2masked_idx], voxel_coors[masked_idx].long()
            ), "The mapping from decoder to masked input is invalid"

            assert torch.allclose(
                decoder_feats[dec2unmasked_idx], unmasked_voxels
            ), "The mapping from decoder to encoder input is invalid"
            assert torch.allclose(
                decoder_coors[dec2unmasked_idx], unmasked_voxel_coors.long()
            ), "The mapping from decoder to encoder input is invalid"

            if self.use_fake_voxels:
                assert (
                    decoder_feats[dec2fake_idx] == 0
                ).all(), "The mapping from decoder to fake voxels is invalid"
                assert torch.allclose(
                    decoder_coors[dec2fake_idx], fake_voxel_coors.long()
                ), "The mapping from decoder to fake voxels is invalid"

        return voxel_info_decoder, voxel_info_encoder

    def get_voxel_indices(self, coors):
        vx, vy, vz = self.sparse_shape
        indices = (
            coors[:, 0] * vz * vy * vx  # batch
            + coors[:, 1] * vy * vx  # z
            + coors[:, 2] * vx  # y
            + coors[:, 3]  # x
        ).long()
        return indices

    def get_voxel_coors(self, indices):
        vx, vy, vz = self.sparse_shape
        coors = torch.zeros((len(indices), 4), device=indices.device, dtype=torch.int32)
        coors[:, 0] = indices // (vx * vy * vz)
        coors[:, 1] = (indices % (vx * vy * vz)) // (vx * vy)
        coors[:, 2] = (indices % (vx * vy)) // vx
        coors[:, 3] = indices % vx
        return coors

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="SubMConv3d"),
    ):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels


@MIDDLE_ENCODERS.register_module()
class SparseEncoder(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(
        self,
        in_channels,
        sparse_shape,
        order=("conv", "norm", "act"),
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        base_channels=16,
        output_channels=128,
        encoder_channels=((16,), (32, 32, 32), (64, 64, 64), (64, 64, 64)),
        encoder_paddings=((1,), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1)),
        block_type="conv_module",
    ):
        super().__init__()
        assert block_type in ["conv_module", "basicblock"]
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {"conv", "norm", "act"}

        if self.order[0] != "conv":  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
                order=("conv",),
            )
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key="subm1",
                conv_type="SubMConv3d",
            )

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule, norm_cfg, self.base_channels, block_type=block_type
        )

        self.conv_out = make_sparse_convmodule(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down2",
            conv_type="SparseConv3d",
        )

    @force_fp32(apply_to=("voxel_features"), out_fp16=True)
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features

    def make_encoder_layers(
        self,
        make_block,
        norm_cfg,
        in_channels,
        block_type="conv_module",
        conv_cfg=dict(type="SubMConv3d"),
    ):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str): Type of the block to use. Defaults to
                'conv_module'.
            conv_cfg (dict): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ["conv_module", "basicblock"]
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == "conv_module":
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f"spconv{i + 1}",
                            conv_type="SparseConv3d",
                        )
                    )
                elif block_type == "basicblock":
                    if j == len(blocks) - 1 and i != len(self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f"spconv{i + 1}",
                                conv_type="SparseConv3d",
                            )
                        )
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                            )
                        )
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f"subm{i + 1}",
                            conv_type="SubMConv3d",
                        )
                    )
                in_channels = out_channels
            stage_name = f"encoder_layer{i + 1}"
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
