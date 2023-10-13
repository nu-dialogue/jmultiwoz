#!/bin/bash
#PJM -L rscgrp=cx-single
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o inference-t5-large-olen512-e2e.log

module load gcc/10.3.0
module load cuda/11.6.2
module load cudnn/8.3.3
module load openmpi_cuda/4.1.2
module load nccl/2.12.7

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_API_KEY="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"

. .venv/bin/activate

# T5
python inference.py \
    --tod_model_type "t5" \
    --model_name_or_path "t5/output/t5-large-bs32-ep5-olen256/checkpoints" \
    --max_context_turns 0 \
    --max_output_length 512 \
    --output_dir "output/t5-large-bs32-ep5-olen512" \
    --task_name "e2e" \
    --world_size 4

# OpenAI Zero-Shot
python inference.py \
    --tod_model_type "openai-zs" \
    --model_name_or_path "gpt-4" \
    --max_context_turns 5 \
    --max_output_length 256 \
    --output_dir "output/gpt4-zs-olen256" \
    --task_name "e2e" \
    --world_size 2 

# OpenAI Few-Shot
# python inference.py \
#     --tod_model_type "openai-fs" \
#     --model_name_or_path "gpt-3.5-turbo" \
#     --max_context_turns 5 \
#     --max_output_length 256 \
#     --output_dir "output/gpt3.5-fs-olen256" \
#     --task_name "e2e" \
#     --world_size 2
