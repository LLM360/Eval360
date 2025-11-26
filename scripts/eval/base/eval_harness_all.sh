#!/bin/bash

# to run all eval tasks, we need 8 nodes in total
# MODEL_NAME="k2plus_stage2_attn64k_jais250k_tp8_normal"
# MODEL_NAME="k2plus_stage2.5_attn32k_jais250k_tp8"
# MODEL_NAME="k2plus_stage3_attn128k_jais250k_tp8_bestfit"
# MODEL_NAME="k2plus_stage3_attn128k_jais250k_rope10m_tp8_bestfit"
# MODEL_NAME="k2plus_stage4_attn512k_jais250k_tp8_bestfit_400nodes_new"
# MODEL_NAME="k2plus_stage5_attn32k_jais250k_rope10m_tp8"
# MODEL_NAME="k2plus_stage5_attn32k_jais250k_rope10m_tp8_with_think_separated"
# MODEL_NAME="k2plus_stage5_attn32k_jais250k_rope10m_tp8_with_think_merged"
MODEL_NAME="k2plus_stage4_attn512k_jais250k_rope10m_tp8_bestfit"
# MODEL_NAME="k2plus_stage4_cont_attn512k_jais250k_rope10m_tp8_bestfit"
START_ITER=8000
END_ITER=10000
STEP_SIZE=500

cd /lustrefs/users/runner/workspace/code/Eval360/scripts/eval/base
sbatch ./eval_arc_bbh_piqa.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_hellaswag.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_humaneval_mbpp.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_mmlu.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_mmlu_pro.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_truthfulqa_winogrande_ifeval.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_mmlu_arabic.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_gsm8k_math.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE
sbatch ./eval_gpqa_diamond_gen.sh $MODEL_NAME $START_ITER $END_ITER $STEP_SIZE