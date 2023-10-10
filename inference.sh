#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o inference-base-bs32-ep5-len512-rg.log

module load gcc/11.3.0
module load cuda/11.7.1
module load openmpi_cuda/4.1.5
module load cudnn/8.9.2
module load nccl/2.18.3

export CUDA_VISIBLE_DEVICES="0,1,2,3"

. .venv/bin/activate
python inference.py \
    --tod_model_type "t5" \
    --model_name_or_path "t5/output/t5-base-bs32-ep5-len256/checkpoints" \
    --max_output_length 512 \
    --output_dir "output/t5-base-bs32-ep5-len512" \
    --task_name "rg" \
    --test_file "processed_data/test.json" \
    --world_size 4
