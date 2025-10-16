#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_vllm_serve.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_vllm_serve.err

export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"

MODEL_NAME=/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-72b-instruct
VLLM_PORT=8080

vllm serve ${MODEL_NAME} \
 --tensor_parallel_size 8 \
 --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' \
 --max-model-len 131072  \
 --override-generation-config '{"max_new_tokens": 131072}' \
 --port ${VLLM_PORT}
