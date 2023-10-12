#!/bin/bash
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o train-large-bs32-ep5.log

module load gcc/11.3.0
module load cuda/11.7.1
module load openmpi_cuda/4.1.5
module load cudnn/8.9.2
module load nccl/2.18.3

. ../.venv/bin/activate

export CUDA_VISIBLE_DEVICES="0,1,2,3"

OUTPUT_DIR="output/t5-large-bs32-ep5-len256"
# torchrun --nproc_per_node=4 --nnodes 1 train.py \
mpirun \
    -n 16 \
    -machinefile $PJM_O_NODEINF \
    -display-devel-map \
    -map-by ppr:2:socket \
    python train.py \
        --do_train \
        --do_eval \
        --model_name_or_path "retrieva-jp/t5-large-long" \
        --overwrite_output_dir \
        --output_dir "${OUTPUT_DIR}/checkpoints" \
        --logging_dir "${OUTPUT_DIR}/log" \
        --report_to "tensorboard" \
        --train_file "processed_data/train.json" \
        --validation_file "processed_data/dev.json" \
        --max_target_length 256 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --num_train_epochs 5 \
        --evaluation_strategy "steps" \
        --eval_steps "400" \
        --save_steps "400" \
        --logging_steps "10"
