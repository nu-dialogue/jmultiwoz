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
export OPENAI_ORGANIZATION="org-7zcbVurnSXSKK4Dc7WHyaD8j"
export OPENAI_API_KEY="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"

. .venv/bin/activate

# T5
# python inference.py \
#     --tod_model_type "t5" \
#     --model_name_or_path "tod_models/t5/output/t5-large-bs32-ep5-olen256/checkpoints" \
#     --max_output_length 512 \
#     --output_dir "output/t5-large-bs32-ep5-olen512" \
#     --task_name "e2e" \
#     --world_size 4

# OpenAI Zero-Shot
# python inference.py \
#     --tod_model_type "openai-zs" \
#     --model_name_or_path "gpt-4" \
#     --max_output_length 256 \
#     --output_dir "output/gpt4-zs-olen256" \
#     --task_name "e2e" \
#     --resume_last_run \
#     --world_size 1

# OpenAI Few-Shot
python inference.py \
    --tod_model_type "openai-fs" \
    --model_name_or_path "gpt-4" \
    --max_output_length 256 \
    --output_dir "output/gpt4-fs-olen256" \
    --task_name "e2e" \
    --resume_last_run \
    --world_size 4