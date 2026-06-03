#!/bin/bash
#SBATCH --job-name=Step1
#SBATCH --output=log/step1/step1-%j.out
#SBATCH --error=log/step1/step1-%j.out
#SBATCH --nodes=NUMBER_OF_NODES
#SBATCH --nodelist=YOUR_NODE
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:GPU

module load conda
module load nccl

conda activate sentence-transformers

# Multi-node NCCL settings
export NCCL_TIMEOUT=3600000  # 1 hour timeout (in milliseconds)
export NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1

# Multi-node optimizations
export NCCL_IB_DISABLE=0  # Enable InfiniBand for better inter-node communication
export NCCL_P2P_DISABLE=0  # Enable P2P for better communication
export NCCL_NET_GDR_LEVEL=3  # Enable GPU Direct RDMA if available
export NCCL_SOCKET_IFNAME=^lo,docker0  # Exclude loopback and docker interfaces

# Get master node info
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR"
echo "Node list: $SLURM_NODELIST"
echo "Total nodes: $SLURM_NNODES"
echo "GPUs per node: 8"
echo "Total GPUs: $((SLURM_NNODES * 8))"


model="MODEL FROM HF" 
model_name="model_name"


lr_list=(1e-4 2e-4 1e-5 2e-5)
temp_list=(0.09 0.07 0.05 0.03 0.01)
warmup_list=(0.1)
dataset=("245M")
loss_list=("MultipleNegativesSymmetricRankingLossReweighting") 
cls="False"
optimizer="adamw_torch"
lr_type="CosineLR" # LinearLR, CosineLR, DecayLR, FlatternLR

# echo "xlmr/${model_name}-step1-${loss_list[0]}-lr${lr_list[0]}-temp${temp_list[0]}-warmup${warmup_list[0]}-Data${dataset[0]}-cls${cls}-${optimizer}-${lr_type}-HERO"


for lr in ${lr_list[@]}
    do
    for temp in ${temp_list[@]}
        do
        for warmup in ${warmup_list[@]}
            do
            for loss in ${loss_list[@]}
                do
                for data in ${dataset[@]}
                    do
                    srun torchrun \
                        --nnodes=$SLURM_NNODES \
                        --nproc_per_node=8 \
                        --rdzv_id=$SLURM_JOB_ID \
                        --rdzv_backend=c10d \
                        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
                        step1_pre_training.py \
                        --model $model \
                        --output "e5_only/${model_name}-step1-${loss}-lr${lr}-temp${temp}-warmup${warmup}-Data${data}-cls${cls}-${optimizer}-${lr_type}-HERO" \
                        --max_seq_length 512 \
                        --batch_size 96 \
                        --learning_rate $lr \
                        --cls $cls \
                        --temperature $temp \
                        --warmup_proportion $warmup \
                        --loss_function $loss \
                        --dataset $data \
                        --epochs 1 \
                        --optimizer $optimizer \
                        --mini_batch_size 320 \
                        --gather_across_devices False \
                        --lr_type $lr_type
                done
            done
        done
    done
done
