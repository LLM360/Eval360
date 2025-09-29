#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=higherprio
#SBATCH --output=/lustrefs/users/runner/slurm/eval_longbench.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_longbench.err

export PATH="/lustrefs/users/runner/anaconda3/envs/loom/bin:$PATH"
# MODEL_PATH=Meta-Llama/Meta-Llama-3.1-8B-Instruct
MODEL_PATH=/lustrefs/users/runner/checkpoints/huggingface/iter_1250000
cd /lustrefs/users/runner/workspace/code/Eval360/LOOM-Scope
loom-scope.run \
    --model_path $MODEL_PATH \
    --cfg_path ./benchmarks/General/LongBench/configs/LongBench.yaml \
    --device 0 1 2 3 4 5 6 7 \
    --gp_num 8 \
    --eval \
    --save_tag K2_PLUS_LongBench \
    --max_length 31500
