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

# Define models array: each element contains "model_name endpoint_address"
MODELS=(
    "qwen2.5-72b-instruct    azure-uk-hpc-H200-instance-329:8080"
)
CKPT_DIR="/lustrefs/users/runner/checkpoints/huggingface"

# Define metrics array: each element contains "metric_name:fewshot_count:batch_size"
METRICS=(
    "aime25:0:auto"
    "aime24:0:auto"
    "gpqa_diamond_cot_zeroshot:0:auto"
    # "gsm8k:0:auto"
    # "gsm8k_cot:0:auto"
    # "minerva_math:0:auto"
    "gsm8k_reasoning_instruct:0:auto"
    "minerva_math_reasoning_instruct:0:auto"
    # "humaneval_instruct:0:auto"
    "mbpp_instruct:0:auto"
    # "humaneval_64_instruct:0:auto"
    "mmlu_pro:0:auto"
    "mmlu_redux_generative:0:auto"
    "ruler:0:auto"
    # "truthfulqa:0:1"
    # "winogrande:0:1"
    "ifeval:0:auto"
)

for model_config in "${MODELS[@]}"; do
    # Split the configuration into components
    IFS=' ' read -r MODEL_NAME BASE_ADDR <<< "$model_config"
    echo "Evaluating ${MODEL_NAME} at ${BASE_ADDR}"
    BASE_URL=http://${BASE_ADDR}/v1/chat/completions
    MODEL_CKPT=${CKPT_DIR}/${MODEL_NAME}

    for metric_config in "${METRICS[@]}"; do
        # Split the configuration into metric name and fewshot count
        IFS=':' read -r METRIC_NAME NUM_FEWSHOT BATCH_SIZE <<< "$metric_config"
        echo "Evaluating ${METRIC_NAME} with ${NUM_FEWSHOT} fewshot..."

        # Add generation kwargs
        if [[ "$METRIC_NAME" == *"ruler"* ]]; then
            GEN_KWARGS='--metadata {"max_seq_lengths":[4096,8192,16384,32768,65536,131072]}'
        else
            GEN_KWARGS="--gen_kwargs do_sample=true,temperature=1.0,top_p=0.95,max_gen_toks=32768"
        fi

        lm_eval --model local-chat-completions \
            --model_args pretrained=${MODEL_CKPT},base_url=${BASE_URL},num_concurrent=10,max_retries=2,timeout=3600 \
            --tasks ${METRIC_NAME} \
            --output_path ${MODEL_CKPT}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
            --num_fewshot $NUM_FEWSHOT \
            --batch_size $BATCH_SIZE \
            --log_samples \
            --apply_chat_template \
            --fewshot_as_multiturn \
            --confirm_run_unsafe_code \
            $GEN_KWARGS &
    done
done
sleep infinity
