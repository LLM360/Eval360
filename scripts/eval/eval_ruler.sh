#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_ruler.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_ruler.err

export PATH="/lustrefs/users/runner/anaconda3/envs/loom/bin:$PATH"
# MODEL_PATH=Meta-Llama/Meta-Llama-3.1-8B-Instruct
MODEL_PATH=/lustrefs/users/runner/checkpoints/huggingface/iter_1250000
cd /lustrefs/users/runner/workspace/code/Eval360/LOOM-Scope
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
loom-scope.run \
    --model_path $MODEL_PATH \
    --cfg_path /lustrefs/users/runner/workspace/code/Eval360/LOOM-Scope/benchmarks/General/RULER/configs/RULER.yaml \
    --device 0 1 2 3 4 5 6 7 \
    --gp_num 8 \
    --eval \
    --server vllm \
    --gpu_memory_utilization 0.70 \
    --max_model_len 32768 \
    --max_length 32768 \
    --save_tag K2_PLUS_VLLM_RULER
