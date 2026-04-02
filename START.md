inference

```
python run.py \
    --mode inference \
    --seed 2023 \
    --dataset_name email \
    --context_data_file ../benchmark/email/test.jsonl \
    --attack_data_file ../benchmark/text_attack_test.json \
    --llm_config_file ../config/qwen3_4b.yaml \
    --batch_size 20 \
    --output_path ../output/qwen3_4b_email_response.jsonl \
    --log_steps 10 \
    --resume

python run.py \
    --mode inference \
    --seed 2023 \
    --dataset_name code \
    --context_data_file ../benchmark/code/test.jsonl \
    --attack_data_file ../benchmark/text_attack_test.json \
    --llm_config_file ../config/qwen3_4b.yaml \
    --batch_size 20 \
    --output_path ../output/qwen3_4b_code_response.jsonl \
    --log_steps 10 \
    --resume

```


clean
```
python collect_clean_response.py --seed 2023 --dataset_name email     --context_data_file ../benchmark/email/train.jsonl     --attack_data_file ../benchmark/text_attack_train.json     --llm_config_file ../config/qwen3_4b.yaml     --batch_size 20 --output_path ../output/clean/email_clean.jsonl     --log_steps 10 --resume --split train
```
