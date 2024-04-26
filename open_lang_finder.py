import torch
import os
import matplotlib.pyplot as plt
import math

from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from root_pipeline import RootPipeline, LangSplatRelevancyPipeline
from relevancy_methods import TestPipeline
from bp_methods import OpaScalingParticleBP, OpaScalingSVBP, View3DSVBP

from torch_bp.inference.kernels import RBFMedianKernel

if __name__ == "__main__":
    # Set up command line argument parser
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    root_pipeline = PipelineParams(parser)   
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
    parser.add_argument("--scaling_modifier", type=float, default=1.0)
    args = get_combined_args(parser)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model from dataset
    dataset = model.extract(args)
    pipeline_params = root_pipeline.extract(args)

    # Autoencoder to decode
    ae_weights = torch.load(args.ae_ckpt_path, map_location=device)

    # load pipelines
    root_pipeline = RootPipeline(
        dataset_params=dataset, auto_encoder_weights=ae_weights,
        pipeline_params=pipeline_params, device=device,
        args=args
    ).to(device)
    # relevancy_pipeline = LangSplatRelevancyPipeline(
    #     root_pipeline=root_pipeline,
    #     device = device
    # ).to(device)
    relevancy_pipeline = TestPipeline(
        root_pipeline=root_pipeline
    ).to(device)

    ################ Test Pipelines #################################

    # queries = ['uno','gundam'] #N queries

    # # render with no grad
    # cam = root_pipeline.scene.getTrainCameras()[0] # temporary get 1
    # R = torch.tensor(cam.R[None,...], device=device, dtype=torch.float32, requires_grad=True)
    # t = torch.tensor(cam.T[None,...], device=device, dtype=torch.float32, requires_grad=True)
    # opa_scaling = torch.randn_like(root_pipeline.gaussian.get_xyz[...,-1], requires_grad=True)
    # # opa_scaling_norm = torch.distributions.Normal(opa_scaling.mean(), opa_scaling.cov().clamp(min=1e-3)).cdf(opa_scaling)[...,None]
    # opa_scaling_norm = opa_scaling.sigmoid().unsqueeze(-1)
    # valid_map = relevancy_pipeline(queries[0], R, t, 
    #                                opa_scaling=opa_scaling_norm, 
    #                                pipeline_params=pipeline_params)
    # plt.imshow(valid_map[0].detach().cpu().numpy())
    # plt.show()
    # # # num_imgs = valid_map.shape[0]
    # # row = int(math.ceil(math.sqrt(num_imgs)))
    # # fig, axs = plt.subplots(row, row, figsize=(row*6, row*6))
    # # for i, img in enumerate(valid_map):
    # #     j, k = divmod(i, row)
    # #     axs[j,k].imshow(valid_map[i].detach().cpu().numpy())
    # # plt.show()
    # # valid_map.sum().backward(inputs=(R,t,opa_scaling)) # temp test
    # # print('R grad:', R.grad)
    # # print('t grad:', t.grad)
    # # print('op grad:', opa_scaling)
    # pass


    ################ Test BP Methods #################################

    # queries = ['pikachu']
    # rel = None
    queries = ['yellow','uno']
    rel = [('is',0,1)]
    num_particles = 5
    unary_params={'num_patches':4}
    init_sigma = 1.
    tensor_kwargs={'device':device,'dtype':torch.float32}

    # num_iter = 250
    # jitter_sigma = 1000.
    # render_cycle = 5
    # render_dir = './output/lzson'

    # # solver = OpaScalingParticleBP(
    # #     query=query, num_particles=num_particles,
    # #     root_pipeline=root_pipeline, pipeline_params=pipeline_params,
    # #     unary_params=unary_params, init_sigma=init_sigma,
    # #     tensor_kwargs=tensor_kwargs
    # # )

    # # final_opa_scale = solver.run_solver(num_iters=num_iter, jitter_sigma=jitter_sigma,
    # #                                     render_cycle=render_cycle, render_dir=render_dir)

    num_iter = 250
    msg_pass_per_iter = 1
    render_cycle = 5
    render_dir = './output/lzson'

    solver = OpaScalingSVBP(
        queries=queries, relations=rel, 
        num_particles=num_particles,
        root_pipeline=root_pipeline, pipeline_params=pipeline_params,
        unary_params=unary_params, init_sigma=init_sigma,
        kernel=RBFMedianKernel(sigma=0.2),
        optim_type=torch.optim.Adam,
        optim_kwargs={'lr': 0.1},
        tensor_kwargs=tensor_kwargs)
    
    final_opa_scale = solver.run_solver(num_iters=num_iter, msg_pass_per_iter=msg_pass_per_iter,
                                        render_cycle=render_cycle, render_dir=render_dir)

    # num_iter = 250
    # num_particles = 50
    # msg_pass_per_iter = 1
    # render_cycle = 5
    # render_dir = './output/lzson'

    # solver = View3DSVBP(
    #     query=query, num_particles=num_particles,
    #     root_pipeline=root_pipeline, pipeline_params=pipeline_params,
    #     unary_params={}, init_sigma=init_sigma,
    #     kernel=RBFMedianKernel(sigma=0.2),
    #     optim_type=torch.optim.Adam,
    #     optim_kwargs={'lr': 1.0},
    #     tensor_kwargs=tensor_kwargs)
    
    # particles, weights = solver.run_solver(num_iters=num_iter, msg_pass_per_iter=msg_pass_per_iter,
    #                                        render_cycle=render_cycle, render_dir=render_dir)
    
    pass