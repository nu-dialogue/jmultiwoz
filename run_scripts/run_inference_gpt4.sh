#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o inference-gpt4-fs-e2e.log

module load gcc/10.3.0
module load cuda/11.6.2
module load cudnn/8.3.3
module load openmpi_cuda/4.1.2
module load nccl/2.12.7

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_ORGANIZATION="<YOUR_ORG>"
export OPENAI_API_KEY="<YOUR_API_KEY>"

. .venv/bin/activate

# OpenAI Few-Shot
python inference.py \
    --tod_model_type "openai-fs" \
    --model_name_or_path "gpt-4" \
    --max_output_length 256 \
    --output_dir "output/gpt4-fs" \
    --task_name "e2e" \
    --resume_last_run \
    --world_size 4
