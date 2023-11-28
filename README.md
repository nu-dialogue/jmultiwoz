# jmultiwoz-evaluation
## Requirements
- Python 3.9+
```bash
pip install -r requirements.txt
```

## Dataset Preparation
```bash
cd dataset
unzip JMultiWOZ_1.0.zip
```

## TODModel Preparation
See the `README.md` in each model directory.
- T5: [t5/README.md](t5/README.md)
- OpenAI LLM: [llm/README.md](llm/README.md)

## Evaluation
### 1. Run inference to generate responses
> [!NOTE]
> Exact command we used can be found in `inferece.sh`

```bash
python inference.py \
    --tod_model_type "openai-fs" \
    --model_name_or_path "gpt-4" \
    --max_output_length 256 \
    --output_dir "output/gpt4-fs-olen256" \
    --task_name "e2e" \
    --resume_last_run \
    --world_size 1
```
Generated dialogues will be saved in `output/gpt4-fs-olen256/e2e.inference.json`.

### 2. Evaluate the generated responses
> [!NOTE]
> Exact command we used can be found in `evaluate.sh`

```bash
python evaluate.py \
    --dataset_dpath "dataset/JMultiWOZ_1.0" \
    --inference_output_dpath "output/gpt4-fs-olen256" \
    --task_name "e2e"
```
Resulted scores will be saved in `output/gpt4-fs-olen256/e2e.scores_summary.json`.
