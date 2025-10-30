#!/bin/bash
#SBATCH --job-name=ckpt_convert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/ckpt.out
#SBATCH --error=/lustrefs/users/runner/slurm/ckpt.err

TP=8
TOKENIZER="/lustrefs/users/xuezhe.ma/projects/data/tokenizers/jais250k"
CKPT_DIR="/lustrefs/users/runner/workspace/checkpoints"
MODEL_NAME="k2plus_stage3_attn128k_jais250k_rope10m_tp8_bestfit"
HF_CONFIG="/lustrefs/users/runner/checkpoints/huggingface/vocab_trimmed/iter_1249000"

export PATH="/lustrefs/users/runner/anaconda3/envs/xllm2.7.1/bin:/lustrefs/users/runner/anaconda3/bin:$PATH"
# source activate xllm2.7.1
# echo $PATH
# which python

for ((i = 2500 ; i <= 17500; i += 2500)) ;
do
    ITER=$(printf "%07d" $i)
    NEXT_ITER=$(printf "%07d" $((i+2500)))
    echo CONVERTING $ITER ...
    CURRENT_CKPT="${CKPT_DIR}/xllm/${MODEL_NAME}/checkpoints/checkpoint_${ITER}"
    NEXT_CKPT="${CKPT_DIR}/xllm/${MODEL_NAME}/checkpoints/checkpoint_${NEXT_ITER}"

    while [[ ! -d ${CURRENT_CKPT} ]]; do
        echo "Folder ${CURRENT_CKPT} does not exist. Waiting ..."
        sleep 60
    done
    # while [[ ! -d ${NEXT_CKPT} ]]; do
    #     echo "Folder ${NEXT_CKPT} does not exist. Waiting ..."
    #     sleep 60
    # done

    # echo "Folder ${NEXT_CKPT} exists. Start converting ${ITER} ..."
    HF_CKPT="${CKPT_DIR}/huggingface/${MODEL_NAME}/checkpoints/checkpoint_${ITER}"
    if [[ -d ${HF_CKPT} ]]; then
        echo "Folder ${HF_CKPT} exists. Skip converting ${ITER} ..."
        continue
    else
        echo "Folder ${HF_CKPT} does not exist. Start converting ${ITER} ..."
        mkdir -p ${HF_CKPT}
        cd /lustrefs/users/runner/workspace/code/xllm
        python tools/convert_checkpoint_format.py --mode "fsdp2torch" \
            --torch_dir $CURRENT_CKPT --fsdp_dir $CURRENT_CKPT \
            --model_parallel_size $TP
        python tools/ckpt_convertion_xllm_to_hf.py \
            --xllm_dir $CURRENT_CKPT \
            --save_dir $HF_CKPT \
            --tp $TP \
            --hf_config_name_or_path $HF_CONFIG \
            --hf_tokenizer_name_or_path $TOKENIZER \
            --do_tp_copy_sanity_check False \
            --rope_theta 10000000 \
            --max_position_embeddings 131072
            # --max_position_embeddings 524288
        # remove temp files
        echo "remove temp files in ${CURRENT_CKPT}"
        rm -f ${CURRENT_CKPT}/model.tp*.pt
    fi
done
