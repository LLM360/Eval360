#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_aime.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_aime.err


export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
export HF_ALLOW_CODE_EVAL="1"

MODEL_NAME=$1
START_ITER=$2
END_ITER=$3
STEP_SIZE=$4
CKPT_DIR="/lustrefs/users/runner/workspace/checkpoints"
HF_DIR="${CKPT_DIR}/huggingface/${MODEL_NAME}/checkpoints"
echo $HF_DIR

for ((i = $START_ITER; i <= $END_ITER; i += $STEP_SIZE)) ;
do
  iter=$(printf "%07d" $i)
  echo EVALUATING $iter ...

  CKPT_DIR="${HF_DIR}/checkpoint_${iter}"
  while [[ ! -e ${CKPT_DIR}/done.txt ]]; do
    echo "${CKPT_DIR}/done.txt does not exist. Waiting..."
    sleep 60
  done
  
  # Define metrics array: each element contains "metric_name:fewshot_count"
  METRICS=(
    "aime24:0"
    "aime25:0"
  )
  
  # Iterate through each metric configuration
  for metric_config in "${METRICS[@]}"; do
    # Split the configuration into metric name and fewshot count
    IFS=':' read -r METRIC_NAME NUM_FEWSHOT <<< "$metric_config"
    
    echo "Evaluating ${METRIC_NAME} with ${NUM_FEWSHOT} fewshot..."
    
    # if [[ -d ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots ]]; then
    #   echo "eval results for ${iter} (${METRIC_NAME}) exist. Skipping..."
    # else
    # model_args="{\"pretrained\":\"${CKPT_DIR}\",\"tensor_parallel_size\":\"8\",\"dtype\":\"float32\",\"gpu_memory_utilization\":\"0.8\",\"max_model_len\":131072,\"rope_scaling\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768},\"rope_theta\":1000000}"
      lm_eval --model vllm \
        --model_args pretrained=${CKPT_DIR},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.8,max_model_len=32768 \
        --tasks ${METRIC_NAME} \
        --output_path ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
        --batch_size auto \
        --apply_chat_template \
        --num_fewshot $NUM_FEWSHOT \
        --apply_chat_template \
        --log_samples \
        --gen_kwargs do_sample=true,temperature=0.7,max_gen_toks=32000
    # fi
  done
done
