#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        # Python-side statistics for boundary-guided densification.
        self.boundary_score_accum = torch.empty(0)
        self.boundary_denom = torch.empty(0)

        # Backward-compatible defaults; overwritten in training_setup().
        self.boundary_min_observations = 2
        self.boundary_child_opacity_budget = 0.20

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)
        ).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.boundary_min_observations = max(
            1,
            int(
                getattr(
                    training_args,
                    "boundary_min_observations",
                    getattr(training_args, "boundary_min_view_count", 2),
                )
            ),
        )
        self.boundary_child_opacity_budget = float(
            getattr(training_args, "boundary_child_opacity_budget", 0.20)
        )
        self.boundary_child_opacity_budget = min(
            max(self.boundary_child_opacity_budget, 0.0), 1.0
        )

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.boundary_score_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.boundary_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr
        return None

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        if self.boundary_score_accum.numel() > 0:
            self.boundary_score_accum = self.boundary_score_accum[valid_points_mask]
        if self.boundary_denom.numel() > 0:
            self.boundary_denom = self.boundary_denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.boundary_score_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.boundary_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(
        self,
        grads,
        grad_threshold,
        scene_extent,
        N=2,
        force_mask=None,
        child_scale_shrink=1.0,
        base_grads=None,
        child_min_opacity=0.0,
    ):
        """Split normal candidates and conservatively inject residual children.

        Normal gradient-qualified candidates preserve the original 3DGS/Pixel-GS
        behavior: the parent is replaced by N children.

        Boundary-only candidates are weaker by definition. Their parent is kept,
        while N low-opacity children are added. This avoids destroying a stable
        photometric fit before the extra children have received any gradients.
        """
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_by_gradient = padded_grad >= grad_threshold

        # Parent replacement must be decided from the unboosted Pixel-GS
        # gradient. A boundary soft boost is only an invitation to add residual
        # capacity; it must not silently become a destructive normal split.
        if base_grads is None:
            base_grads = grads
        padded_base_grad = torch.zeros((n_init_points), device="cuda")
        padded_base_grad[: base_grads.shape[0]] = base_grads.squeeze()

        padded_force = torch.zeros((n_init_points), device="cuda", dtype=torch.bool)
        if force_mask is not None:
            padded_force[: force_mask.shape[0]] = force_mask.bool()

        large_mask = (
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent
        )
        normal_mask = (padded_base_grad >= grad_threshold) & large_mask
        residual_mask = (selected_by_gradient | padded_force) & large_mask & (~normal_mask)
        selected_pts_mask = normal_mask | residual_mask

        if selected_pts_mask.sum() == 0:
            return

        selected_force_only = residual_mask[selected_pts_mask]
        parent_scaling = self.get_scaling[selected_pts_mask]

        stds = parent_scaling.repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected_pts_mask].repeat(N, 1)
        )

        # Do not alter normal split children. Apply the additional shrink only to
        # boundary-only residual children.
        per_parent_shrink = torch.ones(
            (selected_pts_mask.sum(), 1),
            device="cuda",
            dtype=parent_scaling.dtype,
        )
        if selected_force_only.any():
            per_parent_shrink[selected_force_only] = max(float(child_scale_shrink), 1.0)
        new_scaling = self.scaling_inverse_activation(
            parent_scaling.repeat(N, 1)
            / (0.8 * N * per_parent_shrink.repeat(N, 1))
        )

        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        # For a force-only parent, keep the parent and allocate only a small
        # opacity budget to all children together. At identical projection the N
        # children jointly contribute approximately budget * parent_alpha.
        if selected_force_only.any():
            parent_alpha = self.get_opacity[selected_pts_mask]
            total_child_alpha = (
                parent_alpha * self.boundary_child_opacity_budget
            ).clamp(min=1e-4, max=1.0 - 1e-4)
            per_child_alpha = 1.0 - torch.pow(
                1.0 - total_child_alpha,
                1.0 / float(N),
            )
            per_child_alpha = per_child_alpha.clamp(
                min=max(float(child_min_opacity), 1e-6),
                max=1.0 - 1e-6,
            )
            per_child_raw = inverse_sigmoid(per_child_alpha)
            repeated_force_only = selected_force_only.repeat(N)
            repeated_child_raw = per_child_raw.repeat(N, 1)
            new_opacity[repeated_force_only] = repeated_child_raw[repeated_force_only]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        # Only normal gradient-qualified parents are replaced. Boundary-only
        # parents remain as a stable base representation until their children
        # learn useful residual detail or are pruned.
        prune_filter = torch.cat(
            (
                normal_mask,
                torch.zeros(
                    N * selected_pts_mask.sum(),
                    device="cuda",
                    dtype=torch.bool,
                ),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        boundary_enabled=False,
        boundary_grad_relax=0.60,
        boundary_score_threshold=0.20,
        boundary_boost_lambda=1.00,
        boundary_boost_max=2.00,
        boundary_split_shrink=1.05,
        boundary_min_view_count=None,
        boundary_min_observations=None,
        boundary_child_opacity_budget=None,
        **_unused_boundary_kwargs,
    ):
        # Compatibility with older/newer train.py variants. Some versions call
        # this value ``boundary_min_view_count`` while the precision patch names
        # it ``boundary_min_observations``. They are semantically identical.
        effective_min_observations = self.boundary_min_observations
        if boundary_min_observations is not None:
            effective_min_observations = max(
                1, int(boundary_min_observations)
            )
        elif boundary_min_view_count is not None:
            effective_min_observations = max(
                1, int(boundary_min_view_count)
            )

        if boundary_child_opacity_budget is not None:
            self.boundary_child_opacity_budget = min(
                max(float(boundary_child_opacity_budget), 0.0), 1.0
            )

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        boundary_force_mask = None
        if (
            boundary_enabled
            and self.boundary_score_accum.shape == self.xyz_gradient_accum.shape
        ):
            boundary_score = self.boundary_score_accum / torch.clamp_min(
                self.boundary_denom, 1.0
            )
            boundary_score[boundary_score.isnan()] = 0.0
            boundary_score = boundary_score.clamp(min=0.0, max=1.0)

            # Require repeated, Gaussian-local high-error observations. A single
            # border view can no longer trigger a destructive density change.
            stable_boundary = (
                self.boundary_denom >= float(effective_min_observations)
            )
            grad_norm = torch.norm(grads, dim=-1, keepdim=True)
            large_gaussian = (
                torch.max(self.get_scaling, dim=1).values
                > self.percent_dense * extent
            ).unsqueeze(-1)
            high_boundary = (
                boundary_score >= float(boundary_score_threshold)
            ) & stable_boundary
            relaxed_grad = grad_norm >= (max_grad * float(boundary_grad_relax))
            boundary_force_mask = (
                high_boundary & relaxed_grad & large_gaussian
            ).squeeze(-1)

            # Soft enhancement is also limited to stable, large boundary points.
            eligible_score = torch.where(
                stable_boundary & large_gaussian,
                boundary_score,
                torch.zeros_like(boundary_score),
            )
            boost = 1.0 + float(boundary_boost_lambda) * eligible_score
            boost = boost.clamp(min=1.0, max=float(boundary_boost_max))
            grads_for_densify = grads * boost
        else:
            grads_for_densify = grads

        self.densify_and_clone(grads_for_densify, max_grad, extent)
        self.densify_and_split(
            grads_for_densify,
            max_grad,
            extent,
            force_mask=boundary_force_mask,
            child_scale_shrink=boundary_split_shrink,
            base_grads=grads,
            child_min_opacity=min_opacity * 1.2,
        )

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, pixels):
        self.xyz_gradient_accum[update_filter] += (
            torch.norm(
                viewspace_point_tensor.grad[update_filter, :2],
                dim=-1,
                keepdim=True,
            )
            * pixels[update_filter]
        )
        self.denom[update_filter] += pixels[update_filter]

    def _project_points_to_screen(self, viewpoint_camera):
        """Project Gaussian centers to image coordinates in Python."""
        xyz = self.get_xyz.detach()
        device = xyz.device
        ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=device)
        xyz_h = torch.cat([xyz, ones], dim=1)

        full_proj_transform = viewpoint_camera.full_proj_transform.to(device)
        world_view_transform = viewpoint_camera.world_view_transform.to(device)

        clip = xyz_h @ full_proj_transform
        w = clip[:, 3:4]
        w = torch.where(w.abs() < 1e-7, torch.full_like(w, 1e-7), w)
        ndc = clip[:, :3] / w

        image_width = float(viewpoint_camera.image_width)
        image_height = float(viewpoint_camera.image_height)
        x = (ndc[:, 0] + 1.0) * 0.5 * image_width
        y = (ndc[:, 1] + 1.0) * 0.5 * image_height

        view = xyz_h @ world_view_transform
        depth = view[:, 2]
        return x, y, depth

    def add_boundary_densification_stats(
        self,
        viewpoint_cam,
        radii,
        visibility_filter,
        pixels,
        rendered_image,
        gt_image,
        scene_extent,
        opt,
    ):
        """Accumulate locally validated boundary-truncation statistics.

        The image-level boundary ratio is only a cheap first-stage filter. The
        final decision is Gaussian-local: each projected center samples a smoothed
        residual map, and only repeatedly high-error, truncated, large Gaussians
        accumulate a score.
        """
        if self.get_xyz.shape[0] == 0:
            return
        if self.boundary_score_accum.shape[0] != self.get_xyz.shape[0]:
            self.boundary_score_accum = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )
            self.boundary_denom = torch.zeros(
                (self.get_xyz.shape[0], 1), device="cuda"
            )

        with torch.no_grad():
            H = int(viewpoint_cam.image_height)
            W = int(viewpoint_cam.image_width)
            if H <= 0 or W <= 0:
                return

            band = max(
                4.0,
                min(float(H), float(W)) * float(opt.boundary_band_ratio),
            )

            residual = torch.mean(torch.abs(rendered_image - gt_image), dim=0)
            yy = torch.arange(
                H, device=residual.device, dtype=residual.dtype
            ).view(H, 1)
            xx = torch.arange(
                W, device=residual.device, dtype=residual.dtype
            ).view(1, W)
            dist_img = torch.minimum(
                torch.minimum(xx, float(W - 1) - xx),
                torch.minimum(yy, float(H - 1) - yy),
            )
            edge_mask = dist_img < band
            if edge_mask.sum() == 0:
                return

            edge_err = residual[edge_mask].mean()
            global_err = residual.mean()
            error_ratio = edge_err / torch.clamp_min(global_err, 1e-6)
            if error_ratio < float(opt.boundary_error_threshold):
                return

            # Smooth the residual before point sampling so a one-pixel mismatch
            # does not create an unstable split decision.
            local_window = max(3, int(getattr(opt, "boundary_local_window", 9)))
            if local_window % 2 == 0:
                local_window += 1
            max_kernel = min(H, W)
            if max_kernel % 2 == 0:
                max_kernel -= 1
            local_window = min(local_window, max(1, max_kernel))
            local_residual_map = F.avg_pool2d(
                residual[None, None],
                kernel_size=local_window,
                stride=1,
                padding=local_window // 2,
            )[0, 0]

            x, y, depth = self._project_points_to_screen(viewpoint_cam)
            r = radii.detach().float()
            px = pixels.detach().view(-1, 1).float()

            dist_to_border = torch.minimum(
                torch.minimum(x, float(W - 1) - x),
                torch.minimum(y, float(H - 1) - y),
            )
            truncation = torch.clamp(
                (r - dist_to_border) / torch.clamp_min(r, 1e-6),
                0.0,
                1.0,
            )
            boundary_band_weight = torch.clamp(
                (band - dist_to_border) / band,
                0.0,
                1.0,
            )

            depth_threshold = torch.tensor(
                float(opt.depth_threshold) * float(scene_extent),
                device=depth.device,
                dtype=depth.dtype,
            )
            depth_scale = torch.clamp(
                (depth / torch.clamp_min(depth_threshold, 1e-6)) ** 2,
                0.0,
                1.0,
            )

            large_gaussian = (
                torch.max(self.get_scaling, dim=1).values
                > self.percent_dense * scene_extent
            )
            visible = visibility_filter.bool() & (r > 0)

            # For centers outside the image but with a support intersecting it,
            # clamp to the nearest border pixel. This samples the residual of the
            # actually visible part rather than discarding the Gaussian.
            xi = x.round().long().clamp(0, W - 1)
            yi = y.round().long().clamp(0, H - 1)
            local_error = local_residual_map[yi, xi]
            local_error_ratio = local_error / torch.clamp_min(global_err, 1e-6)

            local_threshold = float(
                getattr(opt, "boundary_local_error_threshold", 1.10)
            )
            local_cap = max(
                float(getattr(opt, "boundary_local_error_cap", 1.80)),
                local_threshold + 1e-6,
            )
            local_valid = local_error_ratio >= local_threshold

            valid = (
                visible
                & large_gaussian
                & (truncation >= float(opt.boundary_min_truncation))
                & local_valid
            )
            if valid.sum() == 0:
                return

            px_weight = torch.sqrt(torch.clamp(px.squeeze(-1), min=0.0))
            visible_px_mean = torch.clamp_min(px_weight[visible].mean(), 1e-6)
            px_weight = (px_weight / visible_px_mean).clamp(0.25, 2.0)

            # Threshold-level local residual keeps half weight; increasingly hard
            # local regions approach full weight. This preserves useful LPIPS-
            # oriented detail growth without letting a noisy pixel dominate.
            local_progress = (
                (local_error_ratio - local_threshold)
                / (local_cap - local_threshold)
            ).clamp(0.0, 1.0)
            local_weight = 0.5 + 0.5 * local_progress

            score = (
                truncation
                * boundary_band_weight
                * depth_scale
                * px_weight
                * local_weight
            ).clamp(0.0, 1.0)

            self.boundary_score_accum[valid] += score[valid].unsqueeze(-1)
            self.boundary_denom[valid] += 1.0
