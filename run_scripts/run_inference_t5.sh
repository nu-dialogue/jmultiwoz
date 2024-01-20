#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o inference-t5-large-e2e.log

module load gcc/10.3.0
module load cuda/11.6.2
module load cudnn/8.3.3
module load openmpi_cuda/4.1.2
module load nccl/2.12.7

export CUDA_VISIBLE_DEVICES="0,1,2,3"

. .venv/bin/activate

python inference.py \
    --tod_model_type "t5" \
    --model_name_or_path "nu-dialogue/t5-large-jmultiwoz-e2e" \
    --max_output_length 256 \
    --output_dir "output/t5-large" \
    --task_name "e2e" \
    --world_size 4
