#!/bin/bash -l

#SBATCH -D ./
#SBATCH -J Llama-405B-Online-Inference-TP16-SGL

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1  # Ensure 1 task per node
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --partition=main
#SBATCH --output=/lustrefs/users/runner/slurm/sglang_multinode.out
#SBATCH --error=/lustrefs/users/runner/slurm/sglang_multinode.err

echo "[INFO] Activating environment on node $SLURM_PROCID"
if ! source /lustrefs/users/runner/anaconda3/bin/activate sglang; then
    echo "[ERROR] Failed to activate environment" >&2
    exit 1
fi

# Define parameters
model=/lustrefs/users/runner/workspace/checkpoints/huggingface/k2plus_stage1_attn8k_jais250k_tp8/checkpoints/checkpoint_0012500
tp_size=16

echo "[INFO] Running inference"
echo "[INFO] Model: $model"
echo "[INFO] TP Size: $tp_size"

# Set NCCL initialization address using the hostname of the head node
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Launch the model server on each node using SLURM
srun --ntasks=1 --nodes=1 --exclusive --output="/lustrefs/users/runner/slurm/sglang_multinode_node0.out" \
    --error="/lustrefs/users/runner/slurm/sglang_multinode_node0.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --tp "$tp_size" \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank 0 &

srun --ntasks=1 --nodes=1 --exclusive --output="/lustrefs/users/runner/slurm/sglang_multinode_node1.out" \
    --error="/lustrefs/users/runner/slurm/sglang_multinode_node1.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --tp "$tp_size" \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 2 \
    --node-rank 1 &

# Wait for the NCCL server to be ready on port 30000
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done

echo "[INFO] $HEAD_NODE:30000 is ready to accept connections"

# Keep the script running until the SLURM job times out
wait