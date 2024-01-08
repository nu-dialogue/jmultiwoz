#!/bin/bash
#PJM -L rscgrp=cx-share
#PJM -L elapse=48:00:00
#PJM -j
#PJM -S
#PJM -o create_faiss_db.log

module load gcc/11.3.0
module load cuda/11.7.1

. ../../.venv/bin/activate

python create_faiss_db.py \
    --dataset_dpath "../../dataset/JMultiWOZ_1.0" \
    --output_faiss_db_fprefix "output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20" \
    --context_turns 2 \
    --dialogues_per_domain 20
