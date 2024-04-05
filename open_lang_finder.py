# import numpy as np
import torch
from scene import Scene
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from autoencoder.model import Autoencoder
from eval.openclip_encoder import OpenCLIPNetwork

dataset_name = 'sofa'
ae_ckpt_path = os.path.join('autoencoder/ckpt', dataset_name, "ae_ckpt/best_ckpt.pth")
encoder_hidden_dims = [256, 128, 64, 32, 3]
decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
device = torch.device('cuda:0')

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    args = get_combined_args(parser)

    # load model from dataset
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)

    # load gaussian
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint = os.path.join(args.model_path, 'chkpnt30000.pth')
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, args, mode='test')

    # set background
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Autoencoder to decode
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    # CLIP encoder
    queries = ['pikachu','gundam'] #N queries

    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(queries)

    # render
    with torch.no_grad():
        cam = scene.getTrainCameras()[0]
        output = render(cam, gaussians, pipeline_params, background, args)
        encoded_lang_feat = output['language_feature_image'].permute(1,2,0)
        decoded_lang_feat = model.decode(encoded_lang_feat)
        valid_map = clip_model.get_max_across(decoded_lang_feat[None,...]) # (levels, num_queries, height, width)
        n_head, n_prompt, h, w = valid_map.shape
        print(output['render'].shape, output['language_feature_image'].shape, decoded_lang_feat.shape, valid_map.shape)
        fig, axs = plt.subplots(1,2, figsize=(6,12))
        # axs[0].imshow(output['render'].permute(1,2,0).cpu().numpy())
        axs[0].imshow(valid_map[0,0,...].cpu().numpy())
        axs[1].imshow(output['language_feature_image'].permute(1,2,0).cpu().numpy())
        plt.show()