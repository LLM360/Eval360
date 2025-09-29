#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_ruler_%j.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_ruler_%j.err
#SBATCH --job-name=eval_ruler
#SBATCH --array=0-12
##SBATCH --dependency=afterany:37263_[0-7

export PATH="/lustrefs/users/runner/anaconda3/envs/loom/bin:$PATH"
SAVE_TAG=K2_PLUS_STAGE2_FIX_0012500
# SAVE_TAG=QWEN25-72B
# MODEL_PATH=Meta-Llama/Meta-Llama-3.1-8B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-72B
# MODEL_PATH=/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0135000/
MODEL_PATH=/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage2_attn64k_jais250k_tp8_bestfit_fix/checkpoints/checkpoint_0012500
cd /lustrefs/users/runner/workspace/code/Eval360/LOOM-Scope
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

task_list=("niah_multikey_1" "niah_multikey_2" "niah_multikey_3" "niah_multiquery" "niah_multivalue" "niah_single_1" "niah_single_2" "niah_single_3" "cwe" "fwe" "qa_1" "qa_2" "vt")
task=${task_list[$SLURM_ARRAY_TASK_ID]}
tmp_config=/lustrefs/users/runner/workspace/code/Eval360/LOOM-Scope/benchmarks/General/RULER/configs/tmp_${task}_config.yaml
cat > "$tmp_config" <<EOF
benchmark_name: RULER
task_names: ["$task"]
length: [4096, 8192, 16384, 32768, 65536] #, 131072]
build_data: False
num_samples: 500
no_template_tasks: all
template: default
EOF

loom-scope.run \
    --model_path $MODEL_PATH \
    --cfg_path $tmp_config \
    --device 0 1 2 3 4 5 6 7 \
    --gp_num 8 \
    --eval \
    --server vllm \
    --gpu_memory_utilization 0.70 \
    --max_model_len 131072 \
    --max_length 131072 \
    --save_tag $SAVE_TAG  # K2_PLUS_VLLM_RULER
