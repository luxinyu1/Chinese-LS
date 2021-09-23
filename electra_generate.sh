python bert_generate.py \
    --bert_model 'hfl/chinese-electra-base-generator' \
    --model_cache './model/chinese-electra-base-generator'  \
    --eval_file './dataset/annotation_data.csv' \
    --output_path './data/electra_output.csv'  \
    --max_seq_length 128