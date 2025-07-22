#!/bin/bash

# to run all eval tasks, we need 8 nodes in total
sbatch ./eval_arc_gsm8k.sh
sbatch ./eval_bbh_gpqa_piqa.sh
sbatch ./eval_hellaswag.sh
sbatch ./eval_humaneval_mbpp.sh
sbatch ./eval_mmlu.sh
sbatch ./eval_mmlu_pro.sh
sbatch ./eval_truthfulqa_winogrande_ifeval.sh
sbatch ./eval_mmlu_arabic.sh