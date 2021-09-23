python bert_generate.py \
    --bert_model 'hfl/chinese-bert-wwm-ext' \
    --model_cache './model/chinese_wwm_ext_pytorch'  \
    --eval_file './dataset/annotation_data.csv' \
    --output_path './data/bert_wwm_output.csv'  \
    --max_seq_length 128