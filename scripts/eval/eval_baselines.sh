#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_baseline.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_baseline.err


export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
export HF_ALLOW_CODE_EVAL="1"

declare -A metrics
metrics["mmlu_arabic"]=0
metrics["arc_challenge"]=25
metrics["gsm8k"]=5
metrics["bbh"]=3
metrics["leaderboard_gpqa_diamond"]=0
metrics["hellaswag"]=10
metrics["humaneval"]=0
metrics["mbpp"]=3
metrics["mmlu_pro"]=5
metrics["mmlu"]=5
metrics["truthfulqa"]=0
metrics["winogrande"]=5
metrics["ifeval"]=0
metrics["piqa"]=0
# metrics["social_iqa"]=0
# metrics["race"]=0
# metrics["openbookqa"]=0
metrics["gsm8k_cot"]=8
metrics["minerva_math"]=4
baseline_models=(
  # "/lustrefs/users/runner/checkpoints/huggingface/k2-65b"
  # "/lustrefs/users/runner/checkpoints/huggingface/llama3-70b"
  # "/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-32b"
  # "/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-72b"
  # "/lustrefs/users/runner/checkpoints/huggingface/falcon-h1-34b"
  "/lustrefs/users/runner/checkpoints/huggingface/llama3.1-70b"
  # "/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed/iter_1249000"
  # "/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0135000"
)
for metric_name in ${!metrics[@]}; do
    echo ${metric_name} ${metrics[${metric_name}]}
    for model_path in ${baseline_models[@]}; do
      echo ${model_path}
      if [ ${model_path} == "/lustrefs/users/runner/checkpoints/huggingface/falcon-h1-34b" ]; then
        CUDA_VISIBLE_DEVICES=0,1 lm_eval --model vllm \
          --model_args pretrained=${model_path},tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9 \
          --tasks ${metric_name} \
          --output_path ${model_path}/eval_results/${metric_name}_${metrics[${metric_name}]}shots \
          --batch_size auto \
          --log_samples \
          --confirm_run_unsafe_code
      else
        lm_eval --model vllm \
            --model_args pretrained=${model_path},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.9 \
            --tasks ${metric_name} \
            --output_path ${model_path}/eval_results/${metric_name}_${metrics[${metric_name}]}shots \
            --batch_size auto \
            --log_samples \
            --confirm_run_unsafe_code
      fi
    done
done
