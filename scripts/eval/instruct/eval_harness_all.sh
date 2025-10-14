#!/bin/bash

cd /lustrefs/users/runner/workspace/code/Eval360/scripts/eval
# to run all eval tasks, we need 8 nodes in total
MODELS=(
    # "sft/mid4_sft_instruct:1500:1500:1500"
    # "sft/mid4_sft_reasoning_ot:1500:1500:1500"
    # "sft/mid4_sft_reasoning_am:1500:1500:1500"
    # "sft/mid3_sft:2250:2250:2250"
    # "sft/mid4_sft_instruct_cos_epoch:300:900:300"
    # "sft/mid4_sft_reasoning_am_cos_epoch:750:2250:750"
    # "sft/mid4_sft_reasoning_ot_cos_epoch:2100:6300:2100"
    "sft/mid4.5_sft_reasoning_am_cos_epoch:4500:4500:4500"
    # "sft/mid4_sft_reasoning_oss_cos_epoch:1250:3750:1250"

)
for model_config in "${MODELS[@]}"; do
    # Split the configuration into components
    IFS=':' read -r MODEL_NAME START_ITER END_ITER STEP_SIZE <<< "$model_config"
    echo "Evaluating ${MODEL_NAME} with ${START_ITER} to ${END_ITER} with step size ${STEP_SIZE}"
    # sbatch ./instruct/eval_gsm8k_math.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
    sbatch ./instruct/eval_humaneval_mbpp.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
    # sbatch ./instruct/eval_mmlu_pro.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
    # sbatch ./instruct/eval_mmlu_redux.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
    # sbatch ./instruct/eval_truthfulqa_winogrande_ifeval.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
done