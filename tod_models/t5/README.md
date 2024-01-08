# T5-based Task-oriented Dialogue Modeling
## Prepare Dataset

```bash
python preprocess_dataset.py \
    --dataset_dpath "../../dataset/JMultiWOZ_1.0" \
    --preprocessed_dpath "processed_data"
```

## Training
> [!NOTE]
> Exact command we used can be found in `train.sh`

You can train the model with the following command:

```bash
OUTPUT_DIR="output/t5-base-bs32-ep5-len256"

torchrun --nproc_per_node=4 --nnodes=4 \
    python train.py \
        --do_train \
        --do_eval \
        --model_name_or_path "retrieva-jp/t5-base-long" \
        --overwrite_output_dir \
        --output_dir "${OUTPUT_DIR}/checkpoints" \
        --logging_dir "${OUTPUT_DIR}/log" \
        --report_to "tensorboard" \
        --train_file "processed_data/train.json" \
        --validation_file "processed_data/dev.json" \
        --max_target_length 256 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --num_train_epochs 5 \
        --evaluation_strategy "steps" \
        --eval_steps "400" \
        --save_steps "400" \
        --logging_steps "10"

```
This will output the trained model in `output/t5-base-bs32-ep5-len256/checkpoints`.