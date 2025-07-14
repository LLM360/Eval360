#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=higherprio
#SBATCH --output=/lustrefs/users/runner/slurm/eval_ruler.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_ruler.err

export PATH="/lustrefs/users/runner/anaconda3/envs/loom/bin:$PATH"
MODEL_PATH=Meta-Llama/Meta-Llama-3.1-8B-Instruct
loom-scope.run \
    --model_path $MODEL_PATH \
    --cfg_path ./benchmarks/General/RULER/configs/RULER.yaml \
    --device 0 1 2 3 4 5 6 7 \
    --gp_num 8 \
    --eval \
    --save_tag Llama-3.1-8B-Instruct_RULER
