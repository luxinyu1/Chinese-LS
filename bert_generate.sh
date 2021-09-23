python bert_generate.py \
    --bert_model 'bert-base-chinese' \
    --model_cache './model/bert-base-chinese/'  \
    --eval_file './dataset/annotation_data.csv' \
    --output_path './data/bert_output.csv'  \
    --max_seq_length 128