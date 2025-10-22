#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_harness_all.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_harness_all.err

export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
export HF_ALLOW_CODE_EVAL="1"

# to run all eval tasks, we need 8 nodes in total
MODEL_NAME="qwen2.5-72b-instruct"
MODEL_CKPT="/lustrefs/users/runner/checkpoints/huggingface/${MODEL_NAME}"
BASE_URL="http://azure-uk-hpc-H200-instance-374:8080/v1/completions"
MAX_GEN_TOKENS=32768

# Define metrics array: each element contains "metric_name:fewshot_count:batch_size:trust_remote_code"
METRICS=(
    "arc_challenge:25:1"
    "bbh:3:auto"
    "leaderboard_gpqa_diamond:0:1"
    "piqa:0:1"
    "gpqa_diamond_cot_zeroshot:0:auto"
    # "gsm8k:5:auto"
    # "gsm8k_cot:8:auto"
    # "minerva_math:4:auto"
    # "gsm8k_reasoning_base:0:auto"
    "minerva_math_reasoning_base:0:auto"
    "hellaswag:10:1"
    "humaneval:0:auto"
    "mbpp:3:auto"
    "humaneval_64:0:auto"
    "mmlu_arabic:0:1"
    "mmlu_pro:5:auto"
    "mmlu:5:1"
    "truthfulqa:0:1"
    "winogrande:5:1"
    "ifeval:0:auto"
    "ruler:0:auto"
)

# Iterate through each metric configuration
for metric_config in "${METRICS[@]}"; do
    # Split the configuration into components
    IFS=':' read -r METRIC_NAME NUM_FEWSHOT BATCH_SIZE <<< "$metric_config"

    echo "Evaluating ${METRIC_NAME} with ${NUM_FEWSHOT} fewshot..."

    # Add generation kwargs
    if [[ "$METRIC_NAME" == *"gsm8k"* || "$METRIC_NAME" == *"minerva_math"* || "$METRIC_NAME" == *"gpqa_diamond"* ]]; then
        GEN_KWARGS="--gen_kwargs do_sample=true,temperature=0.7,max_gen_toks=${MAX_GEN_TOKENS}"
    elif [[ "$METRIC_NAME" == *"ruler"* ]]; then
        GEN_KWARGS='--metadata {"max_seq_lengths":[4096,8192,16384,32768,65536,131072]} --gen_kwargs max_gen_toks=${MAX_GEN_TOKENS}'
    else
        GEN_KWARGS=""
    fi

    # Build model args based on trust_remote_code setting
    if [[ "$METRIC_NAME" == *"bbh"* ]]; then
        MODEL_ARGS="pretrained=${MODEL_CKPT},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.7,trust_remote_code=True"
        TRUST_FLAG="--trust_remote_code"
    else
        MODEL_ARGS="pretrained=${MODEL_CKPT},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.7,max_gen_toks=${MAX_GEN_TOKENS}"
        TRUST_FLAG=""
    fi

    lm_eval --model local-completions \
        --model_args pretrained=${MODEL_CKPT},base_url=${BASE_URL},num_concurrent=10,max_retries=2,timeout=3600,tokenized_requests=False,max_gen_toks=${MAX_GEN_TOKENS} \
        --tasks ${METRIC_NAME} \
        --output_path ${MODEL_CKPT}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
        --batch_size $BATCH_SIZE \
        --num_fewshot $NUM_FEWSHOT \
        --log_samples \
        --confirm_run_unsafe_code \
        $TRUST_FLAG \
        $GEN_KWARGS &
done
wait
