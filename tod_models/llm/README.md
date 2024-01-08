# Large Language Model (LLM)-based Task-oriented Dialogue Modeling

Build Faiss Vector Index for retrieving few-shot examples.

```bash
python create_faiss_db.py \
    --dataset_dpath "../../dataset/JMultiWOZ_1.0" \
    --output_faiss_db_fprefix "output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20" \
    --context_turns 2 \
    --dialogues_per_domain 20
```
