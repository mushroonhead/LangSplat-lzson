import torch
from scene import Scene
import os
import math

from arguments import PipelineParams, GroupParams, Namespace
from gaussian_renderer import GaussianModel
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import eval_sh
from utils.spatial_tensor_utils import rot2quat, quat2rot, quat_mult, getWorld2View2


def scaleRot2covar(rot: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Converts rot and scale to 3D covar using RSS^tR^t
    - Inputs:
        - rot: (B,4) tensor, rot in quaternions rxyz
        - scale: (B,3) tensor, scale for xyz
    """
    scale = scale.diag() # make diag to ensure the eig vals are +ve
    rot = quat2rot(rot)

    return rot @ scale @ scale.transpose(-1,-2) @ rot.transpose(-1,-2)


class ImageRelevancyPipeline(torch.nn.Module):
    """
    """
    def __init__(self, 
                 dataset_params: GroupParams, auto_encoder_weights: dict,
                 device: torch.device,
                 args: Namespace) -> None:
        super().__init__()
        # init gaussian and attached scene details
        self.gaussian = GaussianModel(dataset_params.sh_degree)
        self.scene = Scene(dataset_params, self.gaussian, shuffle=False)
        (model_params, first_iter) = torch.load(os.path.join(args.model_path, 'chkpnt30000.pth'))
        self.gaussian.restore(model_params, args, mode='test')
        # set default camera properties (assume first of train set for now)
        try:
            cam = self.scene.getTrainCameras()[0]
        except:
            cam = self.scene.getTestCameras()[0]
        self.tanfovx = math.tan(cam.FoVx * 0.5)
        self.tanfovy = math.tan(cam.FoVy * 0.5)
        self.proj_mat = getProjectionMatrix(znear=0.01, zfar=100., fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).to(device=device)
        self.image_height = cam.image_height
        self.image_width = cam.image_width
        # gaussian background color
        self.background = torch.tensor(([1,1,1] if dataset_params.white_background else [0, 0, 0]), 
                                       dtype=torch.float32, device="cuda")
        # autoencoder to decode
        self.autoencoder = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
        self.autoencoder.load_state_dict(auto_encoder_weights)
        self.autoencoder.eval()
        # CLIP encoder
        self.clip_model = OpenCLIPNetwork(device)

    def forward(self, query: str, R: torch.Tensor, t: torch.Tensor,
                pipeline_params: PipelineParams,
                scaling_modifier=1., override_color = None,
                decode_batchsize=1):
        # set query
        self.clip_model.set_positives([query])
        # render gaussian and keep features only
        _, encoded_lang_feat, _, _ = self.render_gaussian(R, t,
                                                          pipeline_params=pipeline_params,
                                                          scaling_modifier=scaling_modifier, override_color=override_color)
        # split decoding of datasedue to la dimension
        all_valid_map = []
        full_batch_size = encoded_lang_feat.shape[0]
        for start_ind in range(0,full_batch_size,decode_batchsize):
            end_ind = min(start_ind+decode_batchsize, full_batch_size)
            decoded_lang_feat = self.autoencoder.decode(encoded_lang_feat[start_ind:end_ind,...])
            if decoded_lang_feat.dim == 3:
                decoded_lang_feat = decoded_lang_feat[None,...]
            all_valid_map.append(self.clip_model.get_max_across(decoded_lang_feat)) # (levels, 1, height, width)

        # return valid map
        return torch.cat(all_valid_map, dim=0)


    def render_gaussian(self, R: torch.Tensor, t: torch.Tensor,
                        pipeline_params: PipelineParams,
                        scaling_modifier=1., override_color = None):
        """
        Original render function does not propagate gradients for R and t
        Hence we instead render a view from R=I, t=(0,0,0) and rotate gaussians such that
        new_rot = invR @ old_rot, new_3dmean = invR @ old_mean - invR @ t 
        """
        # pin reference
        gaussian = self.gaussian

        # calculate view mat and cam center
        view_mat, cam_center = getWorld2View2(torch.eye(3, device=R.device)[None,...], torch.zeros_like(t))
        view_mat = view_mat.transpose(-1,-2)

        all_rendered_image = []
        all_language_feature_image = []
        all_radii = []
        all_screenspace_points = []

        for i, (view_mat_i, cam_center_i) in enumerate(zip(view_mat, cam_center)): # only 1 rendering pipeline each time
            # generate raster settings
            raster_settings = GaussianRasterizationSettings(
                image_height=int(self.image_height),
                image_width=int(self.image_width),
                tanfovx=self.tanfovx,
                tanfovy=self.tanfovy,
                bg=self.background,
                scale_modifier=scaling_modifier,
                viewmatrix=view_mat_i,
                projmatrix=self.proj_mat,
                sh_degree=gaussian.active_sh_degree,
                campos=cam_center_i,
                prefiltered=False,
                debug=pipeline_params.debug,
                include_feature=True,
            )
            # init rasterizer
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            # process storage for data types
            inv_R = R[i,...].transpose(-1,-2)
            inv_trans = -inv_R @ t[i,...]
            means3D = (inv_R @ gaussian.get_xyz[...,None]).squeeze(-1) + inv_trans
            screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, 
                                                  requires_grad=True, device="cuda") + 0
            try:
                screenspace_points.retain_grad()
            except:
                pass
            means2D = screenspace_points
            opacity = self.gaussian.get_opacity

            # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
            # scaling / rotation by the rasterizer.
            inv_R_quat = rot2quat(inv_R)
            scales = None
            rotations = None
            cov3D_precomp = None
            if pipeline_params.compute_cov3D_python:
                rotations = scaleRot2covar(quat_mult(gaussian.get_rotation, inv_R_quat), gaussian.get_scaling)
            else:
                scales = gaussian.get_scaling
                rotations = quat_mult(gaussian.get_rotation, inv_R_quat)

            # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = None
            colors_precomp = None
            if override_color is None:
                if pipeline_params.convert_SHs_python:
                    shs_view = gaussian.get_features.transpose(1, 2).view(-1, 3, (gaussian.max_sh_degree+1)**2)
                    dir_pp = (gaussian.get_xyz - cam_center_i.repeat(gaussian.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(gaussian.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    shs = gaussian.get_features
            else:
                colors_precomp = override_color

            # features
            language_feature_precomp = gaussian.get_language_feature
            language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
                
            # Rasterize visible Gaussians to image, obtain their radii (on screen).
            rendered_image, language_feature_image, radii = rasterizer(
                means3D = means3D.float(),
                means2D = means2D.float(),
                shs = shs.float(),
                colors_precomp = colors_precomp,
                language_feature_precomp = language_feature_precomp.float(),
                opacities = opacity.float(),
                scales = scales.float(),
                rotations = rotations.float(),
                cov3D_precomp = cov3D_precomp)
            
            all_rendered_image.append(rendered_image)
            all_language_feature_image.append(language_feature_image)
            all_radii.append(radii)
            all_screenspace_points.append(screenspace_points)

        return (torch.stack(all_rendered_image).permute(0,2,3,1), 
                torch.stack(all_language_feature_image).permute(0,2,3,1), 
                torch.stack(all_radii),
                torch.stack(all_screenspace_points))