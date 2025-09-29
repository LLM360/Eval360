#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_mmlu_llmfoudry.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_mmlu_llmfoudry.err
#SBATCH --exclude=azure-uk-hpc-H200-instance-035

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MASTER_PORT=19963
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
echo "Master node address: $MASTER_ADDR"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO

export RANK=$NNODES
export WORLD_SIZE=$WORLD_SIZE
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export LOCAL_WORLD_SIZE=$GPUS_PER_NODE
export NUM_NODES=$NNODES

export LAUNCHER="composer --world_size $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

export CMD="eval/eval.py eval/yamls/hf_eval.yaml"

srun \
--container-image=/lustrefs/users/runner/workspace/code/Eval360/mosaicml+llm-foundry+2.7.0_cu128-latest.sqsh \
--container-mounts=/lustrefs/users/runner/workspace/code/Eval360:/mnt/Eval360 \
--container-workdir=/mnt/Eval360 \
bash -c "
cd llm-foundry && 
if [[ $SLURM_PROCID -eq 0 ]]; then 
    pip install -e . 
else
    sleep 10
fi
sleep 10  # Let install complete
cd scripts && 
export NODE_RANK=$SLURM_PROCID && 
$LAUNCHER --node_rank $SLURM_PROCID $CMD"

# srun \
# --container-image=/lustrefs/users/runner/workspace/code/Eval360/mosaicml+llm-foundry+2.7.0_cu128-latest.sqsh \
# --container-mounts=/lustrefs/users/runner/workspace/code/Eval360:/mnt/Eval360 \
# --container-workdir=/mnt/Eval360 \
# bash -c "cd llm-foundry && pip install -e . && cd scripts && export NODE_RANK=$SLURM_PROCID && $LAUNCHER --node_rank $SLURM_PROCID $CMD"


srun -n 1 --exclusive --gres=gpu:8 --partition=main \
--container-image=/mnt/weka/home/runner/nvidia+nemo+25.07.nemotron-nano-v2.sqsh \
--container-mounts=/mnt/weka/home/runner/Megatron-Bridge:/workdir,/mnt/weka/home/runner/MOE_Herorun_Checkpoints/MoEva-HeroRun_617787/iter_0010000:/workdir/megatron_ckpt \
--container-workdir=/workdir \
--pty bash -i

python examples/models/checkpoint_conversion.py export \
    --hf-model /workdir/hf_ckpt \
    --megatron-path /workdir/megatron_ckpt \
    --hf-path /workdir/hf_ckpt