python bert_generate.py \
    --bert_model 'hfl/chinese-roberta-wwm-ext' \
    --model_cache './model/chinese-roberta-wwm-ext/' \
    --eval_file './dataset/annotation_data.csv' \
    --output_path './data/roberta_output.csv'  \
    --max_seq_length 128