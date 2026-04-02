Inference

```
python run.py     --mode inference     --seed 2023     --dataset_name email     --context_data_file ../benchmark/email/test.jsonl     --attack_data_file ../benchmark/text_attack_test.json     --llm_config_file ../config/qwen3_4b.yaml     --batch_size 20     --output_path ../output/qwen3_4b_email_response.jsonl     --log_steps 10     --resume
```
