# Large Language Model (LLM)-based Task-oriented Dialogue Modeling

This directory contains implementations of LLM-based dialogue models for zero-shot and few-shot learning.

The models and the prompts are implemented with reference to the LLM-based pipeline proposed by [Hudeček and Dusek (2023)](https://arxiv.org/abs/2304.06556):
- Paper: Are LLMs All You Need for Task-Oriented Dialogue? [[Link](https://arxiv.org/abs/2304.06556)]
- Official repo: [vojtsek/to-llm-bot](https://github.com/vojtsek/to-llm-bot)

## Setup
You can use the models by setting the environment variable `OPENAI_API_KEY` before executing our evaluation scripts (i.e., [`inference.py`](../../inference.py), [`human_eval.py`](../../human_eval.py)), as described in the [root README](../../README.md).

Note that if you want to use the model in a few-shot setting, you need to create Faiss Vector Index once in advance. Use `create_faiss_db.py` to build the index as follows:
```bash
python create_faiss_db.py \
    --dataset_dpath "../../dataset/JMultiWOZ_1.0" \
    --output_faiss_db_fprefix "output/faiss_db/hf-sup-simcse-ja-large-ctx2-d20" \
    --context_turns 2 \
    --dialogues_per_domain 20
```
