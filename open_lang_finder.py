import torch
import os
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from relevancy_pipeline import ImageRelevancyPipeline

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ae_ckpt_path", type=str, default=None)
    parser.add_argument('--encoder_dims',
                        nargs = '+',
                        type=int,
                        default=[256, 128, 64, 32, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs = '+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 256, 512],
                        )
    args = get_combined_args(parser)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model from dataset
    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)

    # Autoencoder to decode
    ae_weights = torch.load(args.ae_ckpt_path, map_location=device)

    # load pipeline
    pipeline = ImageRelevancyPipeline(
        dataset_params=dataset, auto_encoder_weights=ae_weights,
        device=device,
        args=args
    )

    queries = ['pikachu','gundam'] #N queries

    # render with no grad
    with torch.no_grad():
        cam = pipeline.scene.getTrainCameras()[0] # temporary get 1
        R = torch.as_tensor(cam.R[None,...], device=device, dtype=torch.float32)
        t = torch.as_tensor(cam.T[None,...], device=device, dtype=torch.float32)
        valid_map = pipeline(queries[0], R, t, pipeline_params)
        plt.imshow(valid_map[0,...].squeeze(0).detach().cpu().numpy())
        pass