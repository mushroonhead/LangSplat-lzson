"""
Methods to do relevancy calculation
"""

from numbers import Real
import torch
import numpy as np
from torchvision.transforms.v2 import RandomCrop
from typing import Optional
from torch_bp.graph.factors import UnaryFactor

from arguments import PipelineParams #, GroupParams, Namespace
from root_pipeline import RootPipeline
from simple_knn._C import distCUDA2
from utils.spatial_tensor_utils import rot_cam_look_at, quat_2_rot, transform_inv, rand_rot

# TODO: example with alpha selection
# TODO: rotate around to select mid point

def scaleRot2covar(rot: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Converts rot and scale to 3D covar using RSS^tR^t
    - Inputs:
        - rot: (B,4) tensor, rot in quaternions rxyz
        - scale: (B,3) tensor, scale for xyz
    """
    ep = 1e-4 # round of values that are too small
    scale = scale.clamp(min=ep).diag_embed() # make diag to ensure the eig vals are +ve
    rot = quat_2_rot(rot)

    return rot @ scale @ scale.transpose(-1,-2) @ rot.transpose(-1,-2)

def scale_rot_2_scale_tril(rot: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.diag_embed()
    rot = quat_2_rot(rot)
    H = rot.double() @ rot.double()
    L = torch.linalg.cholesky(H @ H.transpose(-1,-2))

    return L.float()

class TestPipeline(torch.nn.Module):
    """
    
    """
    def __init__(self, root_pipeline: RootPipeline,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.root_pipeline = root_pipeline
        self.create_pose_distb()
        # self.create_gaussian_distb()

    def forward(self, query: str, R: torch.Tensor, t: torch.Tensor,
                pipeline_params: PipelineParams,
                opa_scaling: Optional[torch.Tensor] = None,
                override_color = None,
                decode_batchsize=1):
        return self.cos_similarity_comp(query, R, t, opa_scaling=opa_scaling,
                                        pipeline_params=pipeline_params, override_color=override_color,
                                        decode_batchsize=decode_batchsize)
        # return self.render_random_views(5, opa_scaling, 
        #                                 pipeline_params=pipeline_params, override_color=override_color)
    
    def cos_similarity_comp(self, query: str, R: torch.Tensor, t: torch.Tensor,
                            pipeline_params: PipelineParams,
                            opa_scaling: Optional[torch.Tensor] = None,
                            override_color = None,
                            decode_batchsize=1) -> torch.Tensor:
        # pin reference
        clip_encoder = self.root_pipeline.clip_encoder
        with torch.no_grad():
            text_embbeding = clip_encoder.encode_text(query) #(512,)
        # render gaussian and keep features only
        _, encoded_lang_feat, _, _ = self.root_pipeline(R, t, opa_scaling=opa_scaling,
                                                        pipeline_params=pipeline_params,
                                                        override_color=override_color) #(B,H,W,3)
        # patch sample to reduce memory requirements
        # encoded_lang_feat = self.patch_sample_2d(encoded_lang_feat, num_samples=9, patch_size=256)
        # decoding, allow splitting due to large memory size
        full_batch_size = encoded_lang_feat.shape[:-3]
        encoded_lang_feat = encoded_lang_feat.reshape(-1, *encoded_lang_feat.shape[-3:])
        cos_sim = []
        flattened_dim_len = encoded_lang_feat.shape[0]
        for start_ind in range(0,encoded_lang_feat.shape[0],decode_batchsize):
            end_ind = min(start_ind+decode_batchsize, flattened_dim_len)
            decoded_lang_feat = self.root_pipeline.autoencoder.decode(encoded_lang_feat[start_ind:end_ind,...]).to(text_embbeding.dtype) #(B,H,W,)
            if decoded_lang_feat.dim == 3:
                decoded_lang_feat = decoded_lang_feat.unsqueeze(0) # (B,H,W,512)
            cos_sim.append((decoded_lang_feat[...,None,:] @ text_embbeding[...,None,None,:,None]).squeeze(-1,-2)) # (B,H,W)
        
        output = torch.cat(cos_sim, dim=0)
        return output.view(*full_batch_size, *output.shape[-2:])
    
    def create_pose_distb(self):
        try:
            cams = self.root_pipeline.scene.getTrainCameras()
        except:
            cams = self.root_pipeline.scene.getTestCameras()
        _, t = zip(*[transform_inv(cam.R, cam.T) for cam in cams])
        t = torch.as_tensor(np.stack(t)/2, dtype=torch.float32, device=self.root_pipeline.device)
        jitter: torch.Tensor = distCUDA2(t)
        mix = torch.distributions.Categorical(torch.ones(t.shape[0], device=self.root_pipeline.device))
        comp = torch.distributions.MultivariateNormal(t, jitter.mean() * torch.eye(3, device=self.root_pipeline.device))
        self.pose_distrb = torch.distributions.MixtureSameFamily(mix, comp)
    
    def create_gaussian_distb(self):
        gaussians = self.root_pipeline.gaussian
        gaussian_pos = gaussians.get_xyz
        gaussian_scale_tril = scale_rot_2_scale_tril(gaussians.get_rotation, gaussians.get_scaling)
        mix = torch.distributions.Categorical(torch.ones(gaussian_pos.shape[0], device=self.root_pipeline.device))
        comp = torch.distributions.MultivariateNormal(gaussian_pos, scale_tril=gaussian_scale_tril)
        self.gaussian_distrb = torch.distributions.MixtureSameFamily(mix, comp)
    
    def random_select_views(self, num_views):
        # sample cam poses
        sampled_cam_poses = self.pose_distrb.sample((num_views,))
        # sample pts to render
        size = torch.linalg.norm(self.root_pipeline.gaussian.get_scaling, dim=-1)
        pos = (self.root_pipeline.gaussian.get_xyz)[size > 1e-3]
        sampled_render_pt = (pos.max(dim=0)[0] + pos.min(dim=0)[0])/2
        # find rotation to use for rendering
        view_rotations = torch.linalg.inv(rot_cam_look_at(sampled_cam_poses, sampled_render_pt[None,:]))
        return transform_inv(view_rotations, sampled_cam_poses)

    
    def render_random_views(self, num_views,
                            opa_scaling: Optional[torch.Tensor],
                            pipeline_params: PipelineParams,
                            override_color=None):
        Rs, ts = self.random_select_views(num_views)

         # pin reference
        clip_encoder = self.root_pipeline.clip_encoder
        # render gaussian and keep features only
        rgb_img, _, _, _ = self.root_pipeline(Rs, ts, opa_scaling=opa_scaling,
                                              pipeline_params=pipeline_params,
                                              override_color=override_color) #(B,H,W,3)

        return rgb_img
    
    def patch_sample_2d(self, data: torch.Tensor, 
                        num_samples: int, patch_size: int) -> torch.Tensor:

        cropping_fn = RandomCrop(patch_size)
        
        data = data.permute(0,3,1,2)
        patches = torch.cat([cropping_fn(data) for _ in range(num_samples)], dim=0) # no good batching fn yet
        patches = patches.permute(0,2,3,1)

        return patches