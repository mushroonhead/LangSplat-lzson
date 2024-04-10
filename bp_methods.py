"""
Stores all belief propagation methods
"""

import torch
from torchvision.transforms.v2 import RandomCrop
from torchvision.transforms.v2.functional import to_pil_image
from torch_bp.graph.factors import UnaryFactor
from torch_bp.graph import MRFGraph
from torch_bp.bp.particle_bp import ParticleBP
from numbers import Real
from typing import Optional
from pathlib import Path
import warnings
import os
import gc

from scene.cameras import Camera
from root_pipeline import RootPipeline
from arguments import PipelineParams

def gaussian_cdf(samples: torch.Tensor,
                 epsilon: float=1e-4) -> torch.Tensor:
    """
    Converts a sample of data into a gaussian cdf set
    - Inputs:
        - samples: (N,) tensor -inf~inf
        - epsilon: float, small positive val to ensure cov is always +ve
    - Returns:
        - cdf: (N,) tensor, 0~1.0
    """
    return torch.distributions.Normal(samples.mean(), samples.cov().clamp(min=epsilon)).cdf(samples)

def batched_gaussian_cdf(samples: torch.Tensor,
                         epsilon: float=1e-4) -> torch.Tensor:
    """
    Naive loop based gaussian cdf for now (not vamppable)
    - Inputs:
        - samples: (B,N) tensor -inf~inf
        - epsilon: float, small positive val to ensure cov is always +ve
    - Returns:
        - cdf: (B,N) tensor, 0~1.0
    """
    return torch.stack([gaussian_cdf(sample, epsilon) for sample in samples], dim=0)

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
    # reshape
    batch_shape = particles.shape[:-2] 
    K, N = particles.shape[-2:]
    particles = particles.view(-1,N) #(B,N)

    # normalize
    scaling = batched_gaussian_cdf(particles, epsilon) #(B,N)
    weights = torch.nn.functional.normalize(weights, dim =-1) #(...,K)

    # shape back
    scaling = scaling.view(*batch_shape,K,N) # (...,K,N)

    # actual prob calculation
    return (scaling * weights[...,None]).sum(-2) # (...,N)


class OpacityScalingUnary(UnaryFactor):
    """
    """
    def __init__(self, query: str,
                 root_pipeline: RootPipeline,
                 pipeline_params: PipelineParams,
                 num_patches : int = 4, patch_size: int = 256,
                 decode_batchsize: int = 1,
                 override_color: Optional[torch.Tensor] = None,
                 beta: Real = 0.5, top_k: int = 64,
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
        opa_scaling_adj = batched_gaussian_cdf(opa_scaling.squeeze(-1)).unsqueeze(-1)
        # render gaussian and keep features only
        _, encoded_lang_feat, _, _ = self.root_pipeline(R, t, opa_scaling=opa_scaling_adj,
                                                        pipeline_params=self.pipeline_params,
                                                        override_color=self.override_color) #(...,H,W,3)
        # for each batch, take n patches
        patches = self.patch_sample_2d(encoded_lang_feat, self.num_patches, self.patch_size)
        # batch cos_similarity comparison
        value_maps = self.cos_similarity_comp(patches) #(...,H,W)
        # final score
        batch_shape = value_maps.shape[:-2]
        score = ((1 - self.beta)*value_maps.mean(dim=(-1,-2)) + # mean of the entire map
                 self.beta*value_maps.view(*batch_shape, -1).topk(self.top_k, dim=-1, sorted=False)[0].mean(dim=-1)) # promote better performing 
                                                                                                      # pixels to perform better

        return score.mean(dim=-1) # mean across views

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

    def cos_similarity_comp(self, patch_features: torch.Tensor) -> torch.Tensor:
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
            val_map.append((decoded_lang_feat[...,None,:] @ self.text_embbeding[...,None,None,:,None]).squeeze(-1,-2)) # (B,H,W)
        
        output = torch.cat(val_map, dim=0)
        return output.view(*full_batch_size, *output.shape[-2:])


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
        self.graph = self.create_graph(
            query=query, 
            root_pipeline=root_pipeline, pipeline_params=pipeline_params,
            unary_params=unary_params, tensor_kwargs=tensor_kwargs)
        # create bp solver
        self.solver = self.create_bp(
            self.graph, root_pipeline=root_pipeline,
            num_particles=num_particles, init_sigma=init_sigma,
            tensor_kwargs=tensor_kwargs)

    def create_graph(self, query: str,
                     root_pipeline: RootPipeline, pipeline_params: PipelineParams,
                     unary_params: dict,
                     tensor_kwargs: dict) -> MRFGraph:
        # graph structure
        edge_ids = []
        unary_factors = [OpacityScalingUnary(
            query, root_pipeline, pipeline_params,
            tensor_kwargs=tensor_kwargs, **unary_params
        )]
        edge_factors = None

        # generate mrf grpah
        graph = MRFGraph(num_nodes=1, edges=edge_ids,
                         edge_factors=edge_factors, unary_factors=unary_factors)
        
        return graph

    def create_bp(self, mrf_graph: MRFGraph,
                  root_pipeline: RootPipeline, 
                  num_particles: int, init_sigma: float,
                  tensor_kwargs: dict) -> ParticleBP:
        # meta data
        num_gaussians = root_pipeline.gaussian.get_xyz.shape[0]

        # particle bp
        init_particles = init_sigma * torch.randn(1, num_particles, num_gaussians, **tensor_kwargs)
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

                rgb_img = self.render_rgb(final_opa_scaling[0]) # only render the first
                pil_img = to_pil_image(rgb_img.permute(2,0,1).detach().cpu())
                pil_img.save(os.path.join(render_dir,f'rgb_{iter:3d}.png'), format='PNG')
            
        return final_opa_scaling
    
    def render_rgb(self, opa_scaling: torch.Tensor) -> torch.Tensor:
        # for now select only the first view
        cam: Camera = self.root_pipeline.scene.getTrainCameras()[0]
        R: torch.Tensor = torch.tensor(cam.R, **self.tensor_kwargs) # (3,3)
        t: torch.Tensor = torch.tensor(cam.T, **self.tensor_kwargs) # (3,)
        rgb_img, _, _, _ = self.root_pipeline(R, t, opa_scaling=opa_scaling,
                                              pipeline_params=self.pipeline_params,
                                              override_color=None) #(H,W,3) # how to override
        return rgb_img