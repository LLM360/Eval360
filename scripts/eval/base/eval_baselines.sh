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
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_ALLOW_CODE_EVAL="1"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# Define metrics and their shot counts
declare -A metrics
# metrics["mmlu_arabic"]=0
# metrics["arc_challenge"]=25
# metrics["gsm8k"]=5
# metrics["bbh"]=3
# metrics["leaderboard_gpqa_diamond"]=0
# metrics["gpqa_diamond_cot_zeroshot"]=0
# metrics["hellaswag"]=10
# metrics["humaneval"]=0
# metrics["mbpp"]=3
# metrics["mmlu_pro"]=5
# metrics["mmlu"]=5
# metrics["truthfulqa"]=0
# metrics["winogrande"]=5
# metrics["ifeval"]=0
# metrics["piqa"]=0
# metrics["gsm8k_cot"]=8
# metrics["minerva_math"]=4
metrics["minerva_math500"]=0
# metrics["humaneval_64"]=0

# Model configurations
single_node_models=(
#   "/lustrefs/users/runner/checkpoints/huggingface/k2-65b"
#   "/lustrefs/users/runner/checkpoints/huggingface/llama3-70b"
#   "/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-32b"
#   "/lustrefs/users/runner/checkpoints/huggingface/qwen2.5-72b"
#   "/lustrefs/users/runner/checkpoints/huggingface/falcon-h1-34b"
  "/lustrefs/users/runner/checkpoints/huggingface/llama3.1-70b"
#   "/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed/iter_1249000"
#   "/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0135000"
#   "/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage2_attn64k_jais250k_tp8_bestfit_fix/checkpoints/checkpoint_0045000"
#   "/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage3_attn128k_jais250k_rope10m_tp8_bestfit/checkpoints/checkpoint_0017500"
#   "/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage4_attn512k_jais250k_rope10m_tp8_bestfit/checkpoints/checkpoint_0010000"
# "/lustrefs/users/runner/checkpoints/huggingface/deepseek-v3.1-base"
)
multi_node_models=(
  # "/lustrefs/users/runner/checkpoints/huggingface/deepseek-v3-base-bf16-new"
)

# Function to check if results already exist
check_results_exist() {
    local model_path="$1"
    local metric_name="$2"
    local shots="$3"
    
    # if [[ -d "${model_path}/eval_results/${metric_name}_${shots}shots" ]]; then
    #     echo "eval results for ${model_path} ${metric_name} ${shots} exist. Skipping..."
    #     return 0
    # else
    #     return 1
    # fi
    return 1
}

# Function to run evaluation for single node models
run_single_node_eval() {
    local model_path="$1"
    local metric_name="$2"
    local shots="$3"
    
    echo "Running evaluation for ${model_path} on ${metric_name} (${shots} shots)"
    
    # Special case for falcon model
    if [[ "${model_path}" == "/lustrefs/users/runner/checkpoints/huggingface/falcon-h1-34b" ]]; then
        if [[ "${metric_name}" == "gpqa_diamond_cot_zeroshot" ]]; then
            # Special configuration for GPQA Diamond CoT with falcon
            CUDA_VISIBLE_DEVICES=0,1 lm_eval --model vllm \
                --model_args pretrained=${model_path},tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9,max_length=32768 \
                --gen_kwargs do_sample=true,temperature=0.7,max_gen_toks=32000 \
                --tasks ${metric_name} \
                --output_path ${model_path}/eval_results/${metric_name}_${shots}shots \
                --batch_size auto \
                --log_samples \
                --num_fewshot $shots \
                --confirm_run_unsafe_code
        else
            # Standard falcon configuration
            CUDA_VISIBLE_DEVICES=0,1 lm_eval --model vllm \
                --model_args pretrained=${model_path},tensor_parallel_size=2,dtype=bfloat16,gpu_memory_utilization=0.9 \
                --tasks ${metric_name} \
                --output_path ${model_path}/eval_results/${metric_name}_${shots}shots \
                --batch_size auto \
                --log_samples \
                --num_fewshot $shots \
                --confirm_run_unsafe_code
        fi
    # Special case for GPQA Diamond CoT Zero-shot with other models
    elif [[ "${metric_name}" == "gpqa_diamond_cot_zeroshot" || "${metric_name}" == "mmlu_generative" || "${metric_name}" == "gsm8k_reasoning_base" || "${metric_name}" == "minerva_math500" ]]; then
        lm_eval --model vllm \
            --model_args pretrained=${model_path},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.9 \
            --gen_kwargs do_sample=true,temperature=1.0,top_p=0.95,max_gen_toks=32768 \
            --tasks ${metric_name} \
            --output_path ${model_path}/eval_results/${metric_name}_${shots}shots \
            --batch_size auto \
            --log_samples \
            --num_fewshot $shots \
            --confirm_run_unsafe_code
    else
        # Standard configuration for other metrics
        lm_eval --model vllm \
            --model_args pretrained=${model_path},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.9 \
            --tasks ${metric_name} \
            --output_path ${model_path}/eval_results/${metric_name}_${shots}shots \
            --batch_size auto \
            --log_samples \
            --num_fewshot $shots \
            --confirm_run_unsafe_code
    fi
}

# Function to run evaluation for multi node models
run_multi_node_eval() {
    local model_path="$1"
    local metric_name="$2"
    local shots="$3"
    
    echo "Running evaluation for ${model_path} on ${metric_name} (${shots} shots)"
    
    lm_eval --model local-completions \
        --model_args model=${model_path},base_url=http://azure-uk-hpc-H200-instance-038:8000/v1/completions,num_concurrent=1,max_retries=3,tokenized_requests=False \
        --tasks ${metric_name} \
        --output_path ${model_path}/eval_results/${metric_name}_${shots}shots \
        --batch_size auto \
        --log_samples \
        --num_fewshot $shots \
        --confirm_run_unsafe_code
}

# Main execution loop
for metric_name in ${!metrics[@]}; do
    echo "Processing metric: ${metric_name} (${metrics[${metric_name}]} shots)"
    
    # Process single node models
    for model_path in ${single_node_models[@]}; do
        echo "Processing model: ${model_path}"
        if ! check_results_exist "${model_path}" "${metric_name}" "${metrics[${metric_name}]}"; then
            run_single_node_eval "${model_path}" "${metric_name}" "${metrics[${metric_name}]}"
        fi
    done
    
    # Process multi node models
    for model_path in ${multi_node_models[@]}; do
        echo "Processing model: ${model_path}"
        if ! check_results_exist "${model_path}" "${metric_name}" "${metrics[${metric_name}]}"; then
            run_multi_node_eval "${model_path}" "${metric_name}" "${metrics[${metric_name}]}"
        fi
    done
done