#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/eval_humaneval_mbpp.out
#SBATCH --error=/lustrefs/users/runner/slurm/eval_humaneval_mbpp.err


export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
export HF_ALLOW_CODE_EVAL="1"

# HF_DIR=/lustrefs/users/runner/checkpoints/huggingface
# HF_DIR=/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed
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
  
  # METRIC_NAME="humaneval_64"
  # NUM_FEWSHOT=0
  # if [[ -d ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots ]]

  # then
  #   echo "eval results for ${iter} exist. Skipping..."
  # else
  #   lm_eval --model vllm \
  #     --model_args pretrained=${CKPT_DIR},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.8 \
  #     --tasks ${METRIC_NAME} \
  #     --output_path ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
  #     --batch_size auto \
  #     --num_fewshot $NUM_FEWSHOT \
  #     --log_samples \
  #     --confirm_run_unsafe_code
  # fi
  METRIC_NAME="humaneval"
  NUM_FEWSHOT=0
  if [[ -d ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots ]]

  then
    echo "eval results for ${iter} exist. Skipping..."
  else
    lm_eval --model vllm \
      --model_args pretrained=${CKPT_DIR},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.8 \
      --tasks ${METRIC_NAME} \
      --output_path ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
      --batch_size auto \
      --num_fewshot $NUM_FEWSHOT \
      --log_samples \
      --confirm_run_unsafe_code
  fi
  METRIC_NAME="mbpp"
  NUM_FEWSHOT=3
  if [[ -d ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots ]]

  then
    echo "eval results for ${iter} exist. Skipping..."
  else
    lm_eval --model vllm \
      --model_args pretrained=${CKPT_DIR},tensor_parallel_size=8,dtype=float32,gpu_memory_utilization=0.8 \
      --tasks ${METRIC_NAME} \
      --output_path ${CKPT_DIR}/eval_results/${METRIC_NAME}_${NUM_FEWSHOT}shots \
      --batch_size auto \
      --num_fewshot $NUM_FEWSHOT \
      --log_samples \
      --confirm_run_unsafe_code
  fi
done
