python bert_generate.py \
    --bert_model 'hfl/chinese-macbert-base' \
    --model_cache './model/chinese-macbert-base'  \
    --eval_file './dataset/annotation_data.csv' \
    --output_path './data/macbert_output.csv'  \
    --max_seq_length 128