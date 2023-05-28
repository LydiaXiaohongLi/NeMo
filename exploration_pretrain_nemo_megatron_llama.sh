#!/bin/bash
set -ex

export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=warn
export NCCL_SOCKET_IFNAME=private
export NCCL_IB_DISABLE=1
export PL_FAULT_TOLERANT_TRAINING=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="10.100.7.239" --master_port=6000 exploration_pretrain_nemo_megatron_llama.py