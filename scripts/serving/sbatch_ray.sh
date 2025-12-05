#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=ray-multi-node-serving
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --partition=main
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --output=/lustrefs/users/runner/slurm/ray.out
#SBATCH --error=/lustrefs/users/runner/slurm/ray.err

set -x
export PATH="/lustrefs/users/runner/anaconda3/bin:$PATH"
export OMPI_MCA_coll_hcoll_enable=0 \
CUDA_DEVICE_ORDER=PCI_BUS_ID \
NCCL_SOCKET_IFNAME=eth0 \
UCX_TLS=rc \
UCX_NET_DEVICES=mlx5_ib0:1 \
NCCL_DEBUG=WARN \
NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml \
NCCL_IB_PCI_RELAXED_ORDERING=1 \
NCCL_IB_QPS_PER_CONNECTION=4 \
NCCL_IGNORE_CPU_AFFINITY=1 \
NCCL_P2P_NET_CHUNKSIZE=$((512 * 1024)) \
NCCL_PXN_DISABLE=1 \
NCCL_MIN_NCHANNELS=32 \
SHARP_SMX_UCX_INTERFACE=mlx5_ib0:1 \
SHARP_COLL_ENABLE_SAT=1 \
SHARP_COLL_LOG_LEVEL=3 \
SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING=1 \
NCCL_COLLNET_ENABLE=1 \
NCCL_IB_HCA=mlx5_ib \
NCCL_IB_TIMEOUT=22

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done
# __doc_worker_ray_end__

# __doc_script_start__
# srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
MODEL_PATH="/lustrefs/users/runner/checkpoints/huggingface/deepseek-v3-base-bf16-new"
vllm serve $MODEL_PATH --tensor-parallel-size 8 --pipeline_parallel_size $SLURM_NNODES --distributed-executor-backend ray

MODEL_PATH="/lustrefs/users/richard.fan/moe_test/fp8"
MODEL_PATH="/lustrefs/users/richard.fan/moe_test/bf16"
DATA_PAR=$((8*$SLURM_NNODES))
VLLM_LOGGING_LEVEL=debug vllm serve $MODEL_PATH --enable-expert-parallel --load-format dummy --data-parallel-size-local 8 --data-parallel-size $DATA_PAR --port 8000 --gpu-memory-utilization 0.90 --no-enable-prefix-caching  --distributed-executor-backend ray --data-parallel-backend ray