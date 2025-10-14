#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_gpqa_diamond.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_gpqa_diamond.err


export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
# export HF_HUB_OFFLINE=1

CKPT_DIR="/lustrefs/users/runner/workspace/checkpoints"
MODEL_NAME=$1
START_ITER=$2
END_ITER=$3
STEP_SIZE=$4
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

  echo "${CKPT_DIR}/done.txt exists. Continuing..."
  # Define metrics array: each element contains "metric_name:fewshot_count:batch_size:trust_remote_code"
  METRICS=(
    "gpqa_diamond_cot_zeroshot:0:auto"
  )
  
  # Iterate through each metric configuration
  for metric_config in "${METRICS[@]}"; do
    # Split the configuration into components
    IFS=':' read -r METRIC_NAME NUM_FEWSHOT BATCH_SIZE <<< "$metric_config"
    
    echo "Evaluating ${METRIC_NAME} with ${NUM_FEWSHOT} fewshot..."
    
    # if [[ -d ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots ]]; then
    #   echo "eval results for ${iter} (${METRIC_NAME}) exist. Skipping..."
    # else
      
      lm_eval --model vllm \
        --model_args pretrained=${CKPT_DIR},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.8,max_length=40000  \
        --tasks ${METRIC_NAME} \
        --output_path ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
        --batch_size $BATCH_SIZE \
        --num_fewshot $NUM_FEWSHOT \
        --apply_chat_template \
        --log_samples \
        --gen_kwargs do_sample=true,temperature=1.0,top_p=0.95,max_gen_toks=32768
    # fi
  done

done
