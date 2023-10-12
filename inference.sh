#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o inference-openai-zs-len256-rg.log

module load gcc/11.3.0
module load cuda/11.7.1
module load openmpi_cuda/4.1.5
module load cudnn/8.9.2
module load nccl/2.18.3

export CUDA_VISIBLE_DEVICES="0,1,2,3"

. .venv/bin/activate

# T5
# python inference.py \
#     --tod_model_type "t5" \
#     --model_name_or_path "t5/output/t5-large-bs32-ep5-len256/checkpoints" \
#     --max_output_length 512 \
#     --output_dir "output/t5-large-bs32-ep5-len512" \
#     --task_name "rg" \
#     --world_size 4

# OpenAI Zero-Shot
OPENAI_API_KEY="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" \
python inference.py \
    --tod_model_type "openai-zs" \
    --model_name_or_path "gpt-3.5-turbo" \
    --max_output_length 256 \
    --output_dir "output_v1/openai-zs-len256" \
    --task_name "rg" \
    --world_size 4

# OpenAI Few-Shot
# ...