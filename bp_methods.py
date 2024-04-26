"""
Stores all belief propagation methods
"""

import numpy as np
import torch
from torchvision.transforms.v2 import RandomCrop
from torchvision.transforms.v2.functional import to_pil_image
from torch_bp.graph.factors import UnaryFactor, PairwiseFactor
from torch_bp.inference.kernels import Kernel
from torch_bp.graph import MRFGraph
from torch_bp.bp.particle_bp import ParticleBP
from torch_bp.bp.svbp import LoopySVBP
from numbers import Real
from typing import Optional, Tuple, Iterable
from pathlib import Path
import warnings
import os
import gc

from scene.cameras import Camera
from root_pipeline import RootPipeline
from arguments import PipelineParams
from simple_knn._C import distCUDA2
from utils.spatial_tensor_utils import transform_inv, rot_cam_look_at, quat_2_rot

def scale_rot_2_scale_tril(rot: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale = scale.diag_embed()
    rot = quat_2_rot(rot)
    H = rot.double() @ rot.double()
    L = torch.linalg.cholesky(H @ H.transpose(-1,-2))

    return L.float()

def gaussian_cdf(samples: torch.Tensor, alpha: float,
                 epsilon: float) -> torch.Tensor:
    """
    Converts a sample of data into a gaussian cdf set
    - Inputs:
        - samples: (N,) tensor -inf~inf
        - epsilon: float, small positive val to ensure cov is always +ve
    - Returns:
        - cdf: (N,) tensor, 0~1.0
    """
    return torch.distributions.Normal(samples.mean(), alpha*samples.cov().clamp(min=epsilon)).cdf(samples)

def batched_gaussian_cdf(samples: torch.Tensor, alpha: float=0.05**2,
                         epsilon: float=1e-4) -> torch.Tensor:
    """
    Naive loop based gaussian cdf for now (not vamppable)
    - Inputs:
        - samples: (B,N) tensor -inf~inf
        - alpha: float, 
        - epsilon: float, small positive val to ensure cov is always +ve
    - Returns:
        - cdf: (B,N) tensor, 0~1.0
    """
    return torch.stack([gaussian_cdf(sample, alpha=alpha, epsilon=epsilon) for sample in samples], dim=0)


def particles_weight_2_final_scale(particles: torch.Tensor, weights: torch.Tensor,
                                   epsilon: float=1e-4) -> torch.Tensor:
    """
    Returns final opa scale for all gaussians, taking into account particle weights
    - Inputs:
        - particles: (...,K,N) tensor -inf~inf
        - weights: (...,K,) tensor -inf~inf
        - epsilon: float, small positive val to ensure cov is always +ve
    - Returns:
        - final_scale: (...,N) tensor, 0~1.0
    """
    # best = weights.argmax()
    # scaling = particles[:, best,...].sigmoid() # batched_gaussian_cdf(particles[:, best,...], alpha=1.0, epsilon=epsilon) #(B,N)

    # return scaling

    # reshape
    batch_shape = particles.shape[:-2] 
    K, N = particles.shape[-2:]
    particles = particles.view(-1,N) #(B,N)

    # normalize
    scaling = particles.sigmoid() # batched_gaussian_cdf(particles, alpha=1.0, epsilon=epsilon) #(B,N)
    weights = torch.nn.functional.softmax(weights, dim =-1) #(...,K)

    # shape back
    scaling = scaling.view(*batch_shape,K,N) # (...,K,N)

    # actual prob calculation
    scaling = (scaling * weights[...,None]).sum(-2) # (...,N)
    return torch.ones_like(scaling) * (scaling > 0.85) # * (scaling > 0.9)

def create_graph(queries: Iterable[str], relations: Optional[Iterable],
                 root_pipeline: RootPipeline, pipeline_params: PipelineParams,
                 unary_params: dict,
                 tensor_kwargs: dict) -> MRFGraph:
    # graph structure
    unary_factors = [OpacityScalingUnary(
        query, root_pipeline, pipeline_params,
        tensor_kwargs=tensor_kwargs, **unary_params
    ) for query in queries]
    if relations is not None:
        edge_ids = [(i,j) for (_,i,j) in relations]
        edge_factors = [OpacityScalingISPairwise(1e3) for (_,i,j) in relations]
    else:
        edge_ids = []
        edge_factors = None

    # generate mrf grpah
    graph = MRFGraph(num_nodes=len(queries), edges=edge_ids,
                     edge_factors=edge_factors, unary_factors=unary_factors)
    
    return graph

def create_graph2(query: str,
                 root_pipeline: RootPipeline, pipeline_params: PipelineParams,
                 unary_params: dict,
                 tensor_kwargs: dict) -> MRFGraph:
    # graph structure
    edge_ids = []
    unary_factors = [Viewpoint3DUnary(
        query, root_pipeline, pipeline_params,
        tensor_kwargs=tensor_kwargs, **unary_params
    )]
    edge_factors = None

    # generate mrf grpah
    graph = MRFGraph(num_nodes=1, edges=edge_ids,
                        edge_factors=edge_factors, unary_factors=unary_factors)
    
    return graph

def render_rgb(opa_scaling: torch.Tensor,
               root_pipeline: RootPipeline, pipeline_params: PipelineParams,
               tensor_kwargs: dict) -> torch.Tensor:
    # for now select only the first view
    cam: Camera = root_pipeline.scene.getTrainCameras()[0]
    R: torch.Tensor = torch.tensor(cam.R, **tensor_kwargs) # (3,3)
    t: torch.Tensor = torch.tensor(cam.T, **tensor_kwargs) # (3,)
    rgb_img, _, _, _ = root_pipeline(R, t, opa_scaling=opa_scaling,
                                        pipeline_params=pipeline_params,
                                        override_color=None) #(H,W,3) # how to override
    return rgb_img


class OpacityScalingUnary(UnaryFactor):
    """
    """
    def __init__(self, query: str,
                 root_pipeline: RootPipeline,
                 pipeline_params: PipelineParams,
                 num_patches : int = 4, patch_size: int = 256,
                 decode_batchsize: int = 1,
                 override_color: Optional[torch.Tensor] = None,
                 beta: Real = 0.5, top_k: int = 256,
                 alpha: Real = 1, 
                 tensor_kwargs = {'device':'cpu', 'dtype':torch.float32}) -> None:
        super().__init__(alpha)
        self.root_pipeline = root_pipeline
        self.pipeline_params = pipeline_params
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.decode_batchsize = decode_batchsize
        self.override_color = override_color
        self.beta = beta
        self.top_k = top_k
        self.tensor_kwargs = tensor_kwargs
        with torch.no_grad():
            self.text_embbeding = root_pipeline.clip_encoder.encode_text(query).to(**tensor_kwargs) #(512,)
            self.neg_embedding = [root_pipeline.clip_encoder.encode_text(neg).to(**tensor_kwargs) for 
                                  neg in ['texture']]


    def log_likelihood(self, opa_scaling: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - opa_scaling: (K,N) tensor
        - Returns:
            - log_prob: (K,) tensor
        """
        # for now select only the first view
        cam: Camera = self.root_pipeline.scene.getTrainCameras()[0]
        R: torch.Tensor = torch.tensor(cam.R, **self.tensor_kwargs) # (3,3)
        t: torch.Tensor = torch.tensor(cam.T, **self.tensor_kwargs) # (3,)
        
        # evaluate on R and t
        return self.full_log_likelihood(R[None,...], t[None,...], opa_scaling)

    def full_log_likelihood(self, R: torch.Tensor, t: torch.Tensor,
                            opa_scaling: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - R: (V..,3,3) tensor
            - t: (V..,3) tensor
            - opa_scaling: (K,N) tensor, single particle
        - Returns:
            - log_prob: (K,) tensor
        """
        # first make incoming opacity scale 
        opa_scaling_adj = opa_scaling.sigmoid().unsqueeze(-1) # batched_gaussian_cdf(opa_scaling.squeeze(-1), alpha=1.0).unsqueeze(-1)
        # render gaussian and keep features only
        _, encoded_lang_feat, _, _ = self.root_pipeline(R, t, opa_scaling=opa_scaling_adj,
                                                        pipeline_params=self.pipeline_params,
                                                        override_color=self.override_color) # (num_particles, height, width, comp_features_dim)
        # for each batch, take n patches
        patches = self.patch_sample_2d(encoded_lang_feat, self.num_patches, self.patch_size) # (num_particles, num_samples, height, width, comp_features_dim)
        # batch cos_similarity comparison
        value_maps = self.cos_similarity_comp(patches, self.text_embbeding) #(num_particles, num_samples, height, width)
        # final score
        batch_shape = value_maps.shape[:-3]
        # score = ((1 - self.beta)*value_maps.mean(dim=(-1,-2,-3)) + # mean of the entire map and across samples
        #          self.beta*value_maps.view(*batch_shape, -1).topk(self.top_k, dim=-1, sorted=False)[0].mean(dim=-1)) # promote better performing 
        #                                                                                               # pixels to perform better
        neg_maps = torch.stack([self.cos_similarity_comp(patches, neg_emb) for neg_emb in self.neg_embedding]).max(dim=0)[0]
        value_maps = torch.nn.functional.softmax(torch.stack([value_maps, neg_maps]), dim=0)[0]
        
        cut_off = value_maps.view(*batch_shape, -1).topk(self.top_k, dim=-1, sorted=False)[0].min(dim=-1)[0] #(num_particles)
        above_cutoff = value_maps >= cut_off[:,None,None,None]
        above_0 = value_maps.abs() > 1e-3
        score = (value_maps * above_cutoff - 1e2 * value_maps.abs() * ~above_cutoff * above_0).sum(dim=(-1,-2,-3))

        return -score

    def full_grad_log_likelihood(self, R: torch.Tensor, t: torch.Tensor,
                                 opa_scaling: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - R: (...,3,3) tensor
            - t: (...,3) tensor
            - opa_scaling: (...,1) tensor
        - Returns:
            - grad_R: (...,3,3) tensor
            - grad_t: (...,3) tensor
            - grad_op: (...,1) tensor
            - log_prob: (...,) tensor
        """
        R = R.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)
        op = opa_scaling.detach().requires_grad_(True)
        unary = self.log_likelihood(R, t, op)
        grad_R, grad_t, grad_op = torch.autograd.grad(unary.sum(), (R,t,op))
        return (self.alpha * grad_R, 
                self.alpha * grad_t,
                self.alpha * grad_op,
                self.alpha * unary)
    
    def patch_sample_2d(self, data: torch.Tensor, 
                        num_samples: int, patch_size: int) -> torch.Tensor:
        """
        Randomly crops certain a small window `num_samples` times for each image
        - Inputs:
            - data: (B,H,W,dim), incoming data to sample from
            - num_samples: int, number of patches to generate
            - patch_size: int, num pixels from height and width
        - Returns:
            - patches: (B,num_samples,H,W,dim)
        """
        cropping_fn = RandomCrop(patch_size)
        
        batch_shape = data.shape[:-3]
        data = data.view(-1, *data.shape[-3:])
        data = data.permute(0,3,1,2)
        patches = torch.stack([cropping_fn(data) for _ in range(num_samples)], dim=-4) # no good batching fn yet
        patches = patches.permute(0,1,3,4,2)
        patches.view(*batch_shape, *patches.shape[-4:])

        return patches

    def cos_similarity_comp(self, patch_features: torch.Tensor,
                            text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Runs 
        """
        # pin reference
        autoencoder = self.root_pipeline.autoencoder
       
        # decoding, allow splitting due to large memory size
        full_batch_size = patch_features.shape[:-3]
        patch_features = patch_features.reshape(-1, *patch_features.shape[-3:]) #(B,H,W,3)
        val_map = []
        flattened_dim_len = patch_features.shape[0]
        for start_ind in range(0,patch_features.shape[0],self.decode_batchsize):
            end_ind = min(start_ind+self.decode_batchsize, flattened_dim_len)
            decoded_lang_feat = autoencoder.decode(patch_features[start_ind:end_ind,...]).to(**self.tensor_kwargs) #(B,H,W,)
            if decoded_lang_feat.dim == 3:
                decoded_lang_feat = decoded_lang_feat.unsqueeze(0) # (B,H,W,512)
            val_map.append((decoded_lang_feat[...,None,:] @ text_embedding[...,None,None,:,None]).squeeze(-1,-2)) # (B,H,W)
        
        output = torch.cat(val_map, dim=0)
        return output.view(*full_batch_size, *output.shape[-2:])
    

class OpacityScalingISPairwise(PairwiseFactor):
    def __init__(self, alpha: Real = 1) -> None:
        super().__init__(alpha)

    def log_likelihood(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x_s: (K1,N) tensor
            - x_t: (K2,N) tensor
        - Returns:
            - eval: (K1,K2) tensor
        """
        N = x_s.shape[-1]
        batch_xs = x_s.shape[:-1]
        batch_xt = x_t.shape[:-1]
        x_s = x_s.view(-1, N)
        x_t = x_t.view(-1, N)

        # temp use something similar to kl div
        x_s = torch.nn.functional.log_softmax(x_s, dim=-1) # (K1,N)
        x_t = torch.nn.functional.log_softmax(x_t, dim=-1) # (K2,N)
        score = (x_s[:,None,...].exp() * (x_s[:,None,:] - x_t[None,:,:])).sum(-1)

        score = score.view(*batch_xs,*batch_xt)
        return score

class OpaScalingParticleBP(object):
    def __init__(self, 
                 query: str, num_particles: int, 
                 root_pipeline: RootPipeline, pipeline_params: PipelineParams,
                 unary_params: dict, init_sigma: float,
                 tensor_kwargs: dict) -> None:
        self.root_pipeline = root_pipeline
        self.pipeline_params = pipeline_params
        self.tensor_kwargs = tensor_kwargs
        # create graph first
        self.graph = create_graph(
            queries=query, 
            root_pipeline=root_pipeline, pipeline_params=pipeline_params,
            unary_params=unary_params, tensor_kwargs=tensor_kwargs)
        # create bp solver
        self.solver = self.create_bp(
            self.graph, root_pipeline=root_pipeline,
            num_particles=num_particles, init_sigma=init_sigma,
            tensor_kwargs=tensor_kwargs)

    def create_bp(self, mrf_graph: MRFGraph,
                  root_pipeline: RootPipeline, 
                  num_particles: int, init_sigma: float,
                  tensor_kwargs: dict) -> ParticleBP:
        # meta data
        num_gaussians = root_pipeline.gaussian.get_xyz.shape[0]

        # particle bp
        init_particles = init_sigma * torch.randn(mrf_graph.N, num_particles, num_gaussians, **tensor_kwargs)
        particle_bp = ParticleBP(init_particles=init_particles, graph=mrf_graph,
                                 sample_mode="nodes", tensor_kwargs=tensor_kwargs)
        
        return particle_bp
    
    def run_solver(self, num_iters: int, jitter_sigma: float, 
                   render_cycle: Optional[int], render_dir: Optional[Path]) -> torch.Tensor:
        # check if need theres location to render
        if render_cycle is not None and render_dir is None:
            warnings.warn('Render dir is not defined even when render cycle is defined. Not image will be rendered!')
            render_cycle = None

        # pass in a memory clearing code
        def clear_mem(solver: ParticleBP, iter: int) -> None:
            if iter % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # solver with render if requested
        if render_cycle is None:
            particles, weights = self.solver.solve(num_iters=num_iters, jitter_sigma=jitter_sigma, 
                                                   return_weights=True, iter_fn=clear_mem)
            final_opa_scaling = particles_weight_2_final_scale(particles, weights)
        else:
            for iter in range(0, num_iters, render_cycle):
                num_iter_this_cycle = min(render_cycle, num_iters - iter)
                particles, weights = self.solver.solve(num_iters=num_iter_this_cycle, jitter_sigma=jitter_sigma, 
                                                       return_weights=True, iter_fn=clear_mem)
                final_opa_scaling = particles_weight_2_final_scale(particles, weights).unsqueeze(-1)

                rgb_img = render_rgb(final_opa_scaling[0],
                                     root_pipeline=self.root_pipeline, pipeline_params=self.pipeline_params, 
                                     tensor_kwargs=self.tensor_kwargs) # only render the first
                pil_img = to_pil_image(rgb_img.permute(2,0,1).detach().cpu())
                pil_img.save(os.path.join(render_dir,f'rgb_{iter:3d}.png'), format='PNG')
            
        return final_opa_scaling
    

class OpaScalingSVBP(object):
    def __init__(self,
                 queries: Iterable[str], relations: Optional[Iterable],  
                 num_particles: int, 
                 root_pipeline: RootPipeline, pipeline_params: PipelineParams,
                 unary_params: dict, init_sigma: float,
                 kernel: Kernel, 
                 optim_type: type, optim_kwargs: dict,
                 tensor_kwargs: dict) -> None:
        self.root_pipeline = root_pipeline
        self.pipeline_params = pipeline_params
        self.tensor_kwargs = tensor_kwargs
        # create graph first
        self.graph = create_graph(
            queries=queries, relations=relations,
            root_pipeline=root_pipeline, pipeline_params=pipeline_params,
            unary_params=unary_params, tensor_kwargs=tensor_kwargs)
        # create bp solver
        self.solver = self.create_bp(
            self.graph, root_pipeline=root_pipeline,
            num_particles=num_particles, init_sigma=init_sigma,
            kernel=kernel,
            optim_type=optim_type, optim_kwargs=optim_kwargs,
            tensor_kwargs=tensor_kwargs)
        
    def create_bp(self, mrf_graph: MRFGraph,
                  root_pipeline: RootPipeline, 
                  num_particles: int, init_sigma: float,
                  kernel: Kernel, 
                  optim_type: type, optim_kwargs: dict,
                  tensor_kwargs: dict) -> LoopySVBP:
        # meta data
        num_gaussians = root_pipeline.gaussian.get_xyz.shape[0]

        # particle bp
        init_particles = init_sigma * torch.randn(mrf_graph.N, num_particles, num_gaussians, **tensor_kwargs)
        solver = LoopySVBP(particles=init_particles, graph=mrf_graph,
                           kernel=kernel, msg_init_mode="uniform",
                           optim_type=optim_type, optim_kwargs=optim_kwargs, 
                           tensor_kwargs=tensor_kwargs)
        
        return solver
    
    def run_solver(self, num_iters: int, msg_pass_per_iter: int,
                   render_cycle: Optional[int], render_dir: Optional[Path]) -> Tuple[torch.Tensor]:
        
        # check if need theres location to render
        if render_cycle is not None and render_dir is None:
            warnings.warn('Render dir is not defined even when render cycle is defined. No image will be rendered!')
            render_cycle = None

        # pass in a memory clearing code
        def clear_mem(solver: LoopySVBP, iter: int) -> None:
            if iter % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # solver with render if requested
        if render_cycle is None:
            particles, weights = self.solver.solve(num_iters=num_iters, msg_pass_per_iter=msg_pass_per_iter,
                                                   iter_fn=clear_mem, return_weights=True)
            final_opa_scaling = particles_weight_2_final_scale(particles, weights)
        else:
            for iter in range(0, num_iters, render_cycle):
                num_iter_this_cycle = min(render_cycle, num_iters - iter)
                particles, weights = self.solver.solve(num_iters=num_iter_this_cycle, msg_pass_per_iter=msg_pass_per_iter, 
                                                       iter_fn=clear_mem, return_weights=True)
                final_opa_scaling = particles_weight_2_final_scale(particles, weights).unsqueeze(-1)

                rgb_img = render_rgb(final_opa_scaling[0],
                                     root_pipeline=self.root_pipeline, pipeline_params=self.pipeline_params, 
                                     tensor_kwargs=self.tensor_kwargs) # only render the first
                pil_img = to_pil_image(rgb_img.permute(2,0,1).detach().cpu())
                pil_img.save(os.path.join(render_dir,f'rgb_{iter:3d}.png'), format='PNG')
            
        return final_opa_scaling
    

class Viewpoint3DUnary(UnaryFactor):
    def __init__(self, query: str,
                 root_pipeline: RootPipeline,
                 pipeline_params: PipelineParams,
                 num_views : int = 4, decode_batchsize: int = 16,
                 alpha: Real = 1,
                 tensor_kwargs = {'device':'cpu', 'dtype':torch.float32}) -> None:
        super().__init__(alpha)
        self.root_pipeline = root_pipeline
        self.pipeline_params = pipeline_params
        self.num_views = num_views
        self.decode_batchsize = decode_batchsize
        self.tensor_kwargs = tensor_kwargs
        with torch.no_grad():
            self.text_embbeding = root_pipeline.clip_encoder.encode_text(query).to(**tensor_kwargs) #(512,)
            # self.neg_embedding = [root_pipeline.clip_encoder.encode_text(neg).to(**tensor_kwargs) for 
            #                       neg in ['texture']]
        self._create_pose_distb()

    def _create_pose_distb(self):
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

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x: (K,3) tensor, xyz position
        - Returns:
            - log_prob: (K,) tensor
        """
        # generate random view
        _, low_feature = self.render_different_views(x)

        # center crop the features
        low_feature = self.center_crop(low_feature)

        # cosine comparisons
        val = self.cos_similarity_comp(low_feature, self.text_embbeding) #(num_particles,num_views,H,W)

        # score calculations
        score = val.sum(dim=(-1,-2,-3))
        
        return score
    
    def generate_random_viewpts(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        - Inputs:
            - x: (K,3) tensor, xyz
        - Returns:
            - R: (K,N,3,3) tensor
            - t: (K,N,3) tensor
        """
        K = x.shape[0]
        # sample cam poses
        sampled_cam_poses = self.pose_distrb.sample((K,self.num_views,)) # (num_particles,num_views,3)
        # find rotation to use for rendering
        view_rotations = torch.linalg.inv(rot_cam_look_at(sampled_cam_poses, x[:,None,:])) # (num_particles,num_views,3,3)
        return transform_inv(view_rotations, sampled_cam_poses)
    
    def render_different_views(self, x: torch.Tensor):
        Rs, ts = self.generate_random_viewpts(x)

        # render gaussian and keep features only
        rgb_img, encoded_lang_feat, _, _ = self.root_pipeline(
            Rs, ts, opa_scaling=None,
            pipeline_params=self.pipeline_params,
            override_color=None) #(B,H,W,3)

        return rgb_img, encoded_lang_feat
    
    def center_crop(self, img: torch.Tensor, size: int) -> torch.Tensor:
        """
        - Inputs:
            - img: (...,H,W,dim) image like object
        - Returns:
            - cropped: (...,size,size,dim) cropped img
        """
        full = img.shape[-3:-1]
        half = torch.tensor(full) * 0.5
        start = (half - 0.5 * size).round().int()
        end = (half + 0.5 * size).round().int()

        return img[...,start[0]:end[0],start[1]:end[1],:]
    
    def cos_similarity_comp(self, patch_features: torch.Tensor,
                            text_embedding: torch.Tensor) -> torch.Tensor:
        """
        Runs 
        """
        # pin reference
        autoencoder = self.root_pipeline.autoencoder
       
        # decoding, allow splitting due to large memory size
        full_batch_size = patch_features.shape[:-3]
        patch_features = patch_features.reshape(-1, *patch_features.shape[-3:]) #(B,H,W,3)
        val_map = []
        flattened_dim_len = patch_features.shape[0]
        for start_ind in range(0,patch_features.shape[0],self.decode_batchsize):
            end_ind = min(start_ind+self.decode_batchsize, flattened_dim_len)
            decoded_lang_feat = autoencoder.decode(patch_features[start_ind:end_ind,...]).to(**self.tensor_kwargs) #(B,H,W,)
            if decoded_lang_feat.dim == 3:
                decoded_lang_feat = decoded_lang_feat.unsqueeze(0) # (B,H,W,512)
            val_map.append((decoded_lang_feat[...,None,:] @ text_embedding[...,None,None,:,None]).squeeze(-1,-2)) # (B,H,W)
        
        output = torch.cat(val_map, dim=0)
        return output.view(*full_batch_size, *output.shape[-2:]) #(num_particles,num_views,H,W)
    

class View3DSVBP(object):
    def __init__(self,
                 query: str, num_particles: int, 
                 root_pipeline: RootPipeline, pipeline_params: PipelineParams,
                 unary_params: dict, init_sigma: float,
                 kernel: Kernel, 
                 optim_type: type, optim_kwargs: dict,
                 tensor_kwargs: dict) -> None:
        self.root_pipeline = root_pipeline
        self.pipeline_params = pipeline_params
        self.tensor_kwargs = tensor_kwargs
        # create graph first
        self.graph = create_graph2(
            query=query, 
            root_pipeline=root_pipeline, pipeline_params=pipeline_params,
            unary_params=unary_params, tensor_kwargs=tensor_kwargs)
        # create distrb of gaussian splats for sampling
        self.create_gaussian_distb()

        # create bp solver
        self.solver = self.create_bp(
            self.graph, root_pipeline=root_pipeline,
            num_particles=num_particles, init_sigma=init_sigma,
            kernel=kernel,
            optim_type=optim_type, optim_kwargs=optim_kwargs,
            tensor_kwargs=tensor_kwargs)

    def create_gaussian_distb(self):
        gaussians = self.root_pipeline.gaussian
        gaussian_pos = gaussians.get_xyz
        gaussian_scale_tril = scale_rot_2_scale_tril(gaussians.get_rotation, gaussians.get_scaling)
        mix = torch.distributions.Categorical(torch.ones(gaussian_pos.shape[0], device=self.root_pipeline.device))
        comp = torch.distributions.MultivariateNormal(gaussian_pos, scale_tril=gaussian_scale_tril)
        self.gaussian_distrb = torch.distributions.MixtureSameFamily(mix, comp)

    def create_bp(self, mrf_graph: MRFGraph,
                  root_pipeline: RootPipeline, 
                  num_particles: int, init_sigma: float,
                  kernel: Kernel, 
                  optim_type: type, optim_kwargs: dict,
                  tensor_kwargs: dict) -> LoopySVBP:
        # particle bp
        init_particles = self.gaussian_distrb.sample((num_particles,)).unsqueeze(0) # node 0 particles
        solver = LoopySVBP(particles=init_particles, graph=mrf_graph,
                           kernel=kernel, msg_init_mode="uniform",
                           optim_type=optim_type, optim_kwargs=optim_kwargs, 
                           tensor_kwargs=tensor_kwargs)
        
        return solver
    
    def run_solver(self, num_iters: int, msg_pass_per_iter: int,
                   render_cycle: Optional[int], render_dir: Optional[Path]) -> Tuple[torch.Tensor]:
        
        # check if need theres location to render
        if render_cycle is not None and render_dir is None:
            warnings.warn('Render dir is not defined even when render cycle is defined. No image will be rendered!')
            render_cycle = None

        # pass in a memory clearing code
        def clear_mem(solver: LoopySVBP, iter: int) -> None:
            if iter % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        # solver with render if requested
        if render_cycle is None:
            particles, weights = self.solver.solve(num_iters=num_iters, msg_pass_per_iter=msg_pass_per_iter,
                                                   iter_fn=clear_mem, return_weights=True)
        else:
            for iter in range(0, num_iters, render_cycle):
                num_iter_this_cycle = min(render_cycle, num_iters - iter)
                particles, weights = self.solver.solve(num_iters=num_iter_this_cycle, msg_pass_per_iter=msg_pass_per_iter, 
                                                       iter_fn=clear_mem, return_weights=True)
                rgb_imgs, _ = self.render_different_views(particles[weights.argmax()])
                pil_img = to_pil_image(rgb_imgs[0].permute(2,0,1).detach().cpu())
                pil_img.save(os.path.join(render_dir,f'rgb_{iter:3d}.png'), format='PNG')
            
        return particles, weights