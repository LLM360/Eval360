#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_ruler_helmet.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_ruler_helmet.err

export PATH="/lustrefs/users/runner/anaconda3/envs/helmet/bin:$PATH"
# MODEL_PATH=Meta-Llama/Meta-Llama-3.1-8B-Instruct
MODEL_PATH=/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed/iter_1249000
cd /lustrefs/users/runner/workspace/code/Eval360/HELMET
for task in ruler ; do
  python eval.py --config configs/${task}.yaml \
    --model_name_or_path $MODEL_PATH \
    --output_dir output/k2-plus-${task} \
    --use_chat_template False
done
