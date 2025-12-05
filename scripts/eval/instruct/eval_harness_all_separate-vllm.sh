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
#SBATCH --exclude=azure-uk-hpc-H200-instance-012,azure-uk-hpc-H200-instance-013

export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
export HF_ALLOW_CODE_EVAL="1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_NAME=allenai/Olmo-3-32B-Think
VLLM_SERVER_ENDPOINT=${VLLM_SERVER_ENDPOINT:-azure-uk-hpc-H200-instance-019}
VLLM_PORT=8000
MAX_GEN_TOKENS=65536
REASONING_EFFORT=
if [ -n "${REASONING_EFFORT}" ]; then
    OUTPUT_PATH=${MODEL_NAME}-${REASONING_EFFORT}/eval_results
else
    OUTPUT_PATH=${MODEL_NAME}/eval_results
fi


SYSTEM_PROMPT="The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The answer are enclosed within \boxed{}. In the answer mention each unknown and its solution, for example, \boxed{10}. Now the user asks you to solve a reasoning problem."

if [ "${VLLM_SERVER_ENDPOINT}" == "localhost" ]; then
    vllm serve ${MODEL_NAME} \
     --tensor_parallel_size 8 \
     --gpu-memory-utilization 0.8 \
     --trust-remote-code \
     --port ${VLLM_PORT} &
fi

until curl -sf ${VLLM_SERVER_ENDPOINT}:${VLLM_PORT}/v1/models; do
    printf '.'
    sleep 5
done

# Define metrics array: each element contains "metric_name:fewshot_count:batch_size"
METRICS=(
    # "aime25:0:auto"
    # "aime24:0:auto"
    # "gpqa_diamond_cot_zeroshot:0:auto"
    # "gsm8k:0:auto"
    # "gsm8k_cot:0:auto"
    # "minerva_math:0:auto"
    # "gsm8k_reasoning_instruct:0:auto"
    # "minerva_math_reasoning_instruct:0:auto"
    "minerva_math500_instruct:0:auto"
    # "humaneval_instruct:0:auto"
    "humaneval_instruct_reasoning:0:auto"
    # "mbpp_instruct:0:auto"
    # "humaneval_64_instruct:0:auto"
    # "mmlu_pro:0:auto"
    # "mmlu_redux_generative:0:auto"
    "truthfulqa:0:auto"
    "winogrande:0:auto"
    # "ifeval:0:auto"
    # "ruler:0:auto"
)

for metric_config in "${METRICS[@]}"; do
    # Split the configuration into metric name and fewshot count
    IFS=':' read -r METRIC_NAME NUM_FEWSHOT BATCH_SIZE <<< "$metric_config"
    echo "Evaluating ${METRIC_NAME} with ${NUM_FEWSHOT} fewshot..."
    
    # Set BASE_URL based on task type
    if [[ "$METRIC_NAME" == "winogrande" || "$METRIC_NAME" == "truthfulqa" ]]; then
        BASE_URL="http://${VLLM_SERVER_ENDPOINT}:${VLLM_PORT}/v1/completions"
        MODEL_TYPE="local-completions"
    else
        BASE_URL="http://${VLLM_SERVER_ENDPOINT}:${VLLM_PORT}/v1/chat/completions"
        MODEL_TYPE="local-chat-completions"
    fi

    # Add generation kwargs
    if [[ "$MODEL_TYPE" == "local-chat-completions" ]]; then
        if [[ "$METRIC_NAME" == *"ruler"* ]]; then
            # GEN_KWARGS='--metadata {"max_seq_lengths":[4096,8192,16384,32768,65536,131072]}'
            GEN_KWARGS='--metadata {"max_seq_lengths":[131072]}'
        else
            GEN_KWARGS='--gen_kwargs do_sample=true,temperature=1.0,top_p=0.95,max_gen_toks=32768'

            # Uncomment the following line to enable different reasoning efforts
            # GEN_KWARGS='--gen_kwargs {"do_sample":true,"temperature":1.0,"top_p":0.95,"max_gen_toks":32768,"chat_template_kwargs":{"reasoning_effort":"'$REASONING_EFFORT'"}}'
            # GEN_KWARGS='--gen_kwargs {"do_sample":true,"temperature":1.0,"top_p":0.95,"max_gen_toks":32768,"chat_template_kwargs":{"thinking":true}}'

        fi
    else
        GEN_KWARGS=''
    fi

    # Build model_args with conditional tokenizer
    MODEL_ARGS="pretrained=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=30,max_retries=2,timeout=5400,max_length=${MAX_GEN_TOKENS}"
    if [[ "$METRIC_NAME" == "winogrande" || "$METRIC_NAME" == "truthfulqa" ]]; then
        MODEL_ARGS="${MODEL_ARGS},tokenizer=${MODEL_NAME}"
    fi

    lm_eval --model ${MODEL_TYPE} \
        --model_args ${MODEL_ARGS} \
        --tasks ${METRIC_NAME} \
        --output_path ${OUTPUT_PATH}/${METRIC_NAME}_${NUM_FEWSHOT}shots \
        --num_fewshot $NUM_FEWSHOT \
        --batch_size $BATCH_SIZE \
        --log_samples \
        --confirm_run_unsafe_code \
        --apply_chat_template --fewshot_as_multiturn \
        $GEN_KWARGS & PIDS+=( $! )

        # Add the following argument if \boxed{} parsing is required
        # --system_instruction "${SYSTEM_PROMPT}" \

done
wait ${PIDS[@]}
