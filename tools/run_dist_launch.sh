set -x

GPUS=$1
RUN_COMMAND=${@:2}
# 如果总gpu数量小于8, 只有一个节点，节点上的gpu数量=总gpu数量;如果gpu总数量大于8, 分配到单个节点上的gpu数量=8
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"29501"}
NODE_RANK=${NODE_RANK:-0}

let "NNODES=GPUS/GPUS_PER_NODE"

python ./tools/launch.py \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --nproc_per_node ${GPUS_PER_NODE} \
    ${RUN_COMMAND}