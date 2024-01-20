export OPENAI_ORGANIZATION="<YOUR_ORG>"
export OPENAI_API_KEY="<YOUR_API_KEY>"

python human_eval.py \
    --tod_model_names gpt4-fs gpt3.5-fs t5-base t5-large \
    --task_ids_fpath task_ids.json \
    --threading_httpd
