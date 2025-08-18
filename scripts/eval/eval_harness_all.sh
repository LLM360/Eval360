#!/bin/bash

# to run all eval tasks, we need 8 nodes in total
# MODEL_NAME="k2plus_stage2_attn64k_jais250k_tp8_normal"
MODEL_NAME="k2plus_stage2_attn64k_jais250k_tp8_bestfit"
cd /lustrefs/users/runner/workspace/code/Eval360/scripts/eval
sbatch ./eval_arc_gsm8k.sh $MODEL_NAME
sbatch ./eval_bbh_gpqa_piqa.sh $MODEL_NAME
sbatch ./eval_hellaswag.sh $MODEL_NAME
sbatch ./eval_humaneval_mbpp.sh $MODEL_NAME
sbatch ./eval_mmlu.sh $MODEL_NAME
sbatch ./eval_mmlu_pro.sh $MODEL_NAME
sbatch ./eval_truthfulqa_winogrande_ifeval.sh $MODEL_NAME
sbatch ./eval_mmlu_arabic.sh $MODEL_NAME
sbatch ./eval_gsm8k_cot_math.sh $MODEL_NAME