#!/bin/bash
CASE_NAME="teatime"

# path to lerf_ovs/label
gt_folder="../data/lerf_ovs/label"

root_path="../"

CUDA_VISIBLE_DEVICES=1 python evaluate_iou_loc.py \
                        --dataset_name ${CASE_NAME} \
                        --feat_dir ${root_path}/output/${CASE_NAME} \
                        --ae_ckpt_dir ${root_path}/autoencoder/ckpt/${CASE_NAME}/best_ckpt.pth \
                        --output_dir ${root_path}/eval_result \
                        --mask_thresh 0.4 \
                        --encoder_dims 256 128 64 32 3 \
                        --decoder_dims 16 32 64 128 256 256 512 \
                        --json_folder ${gt_folder}