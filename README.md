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
1. Run inference to generate responses
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

2. Evaluate the generated responses
> [!NOTE]
> Exact command we used can be found in `evaluate.sh`

```bash
python evaluate.py \
    --dataset_dpath "dataset/JMultiWOZ_1.0" \
    --inference_output_dpath "output/gpt4-fs-olen256" \
    --task_name "e2e"
```
Resulted scores will be saved in `output/gpt4-fs-olen256/e2e.scores_summary.json`.

## Human Evaluation
### 1. Setup web server
Proxy pass must be set to `/dialogue`.

If nginx is used, configure `/etc/nginx/conf.d/default.conf` as follows

```nginx
server {
    listen 80;
    server_name localhost;
    location / {
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_pass http://localhost:8080;
    }
    location /dialogue {
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_pass http://localhost:8080/dialogue;
    }
}
```

### 2. Run web server
```bash
python server.py --tod_model_names gpt4-fs gpt3.5-fs t5-large
```

### 3. Open the web page in your browser
- For local access: http://localhost:8080/dialogue
- For remote access: http://your-server-ip/dialogue
