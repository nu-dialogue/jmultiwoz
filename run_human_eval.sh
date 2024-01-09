
export OPENAI_ORGANIZATION="org-7zcbVurnSXSKK4Dc7WHyaD8j"
export OPENAI_API_KEY="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"

# python human_eval.py --tod_model_names gpt4-fs gpt3.5-fs t5-base t5-large
python human_eval.py \
    --tod_model_names gpt4-fs gpt3.5-fs t5-base t5-large \
    --task_ids_fpath task_ids.json \
    --threading_httpd
