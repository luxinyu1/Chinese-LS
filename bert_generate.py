# This file has been tested in transformers==3.5.0

import logging
import argparse
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer

logger = logging.getLogger(__name__)

def read_eval_dataset(data_path):
    sentences = []
    mask_words = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            row = line.strip().split('\t')
            sentence, mask_word = row[0], row[1]
            sentences.append(''.join(sentence.split(' ')))
            mask_words.append(mask_word)
    return sentences, mask_words

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, value = line.strip().split(',')
            dict[key] = value
    return dict

def encoder(tokenizer, sequence_a, sequence_b, max_length):
    sequence_dict = tokenizer.encode_plus(sequence_a, sequence_b, max_length=max_length, padding=True, return_tensors='pt')
    return sequence_dict

def truncate(sentence, start_index, end_index, window):
    # extract words around the content word
    len_sent = len(sentence)
    len_word = end_index - start_index
    radius = int((window - len_word) / 2)
    word_half_index = int((start_index + end_index) / 2)
    if start_index - radius < 0:
        sentence = sentence[0:window-1]
    elif end_index + radius > len_sent - 1:
        sentence = sentence[len_sent-window-1:len_sent-1]
    else:
        sentence = sentence[start_index-radius:end_index+radius]
    return sentence

def predict_char(tokenizer, model, sentence, mask_sentence, max_length, k):
    sequence_dict = encoder(tokenizer, sentence, mask_sentence, max_length)
    input_ids = sequence_dict['input_ids'].to('cuda')
    attention_masks = sequence_dict['attention_mask'].to('cuda')
    token_type_ids = sequence_dict['token_type_ids'].to('cuda')
    masked_index = int(torch.where(input_ids == tokenizer.mask_token_id)[1][0])
    with torch.no_grad():
        outputs = model(input_ids, attention_masks, token_type_ids) # Return type: tuple(torch.FloatTensor) comprising various elements depending on the configuration (BertConfig) and inputs
    token_logits = outputs[0]
    mask_token_logits = token_logits[0, masked_index, :]
    mask_token_probs = mask_token_logits.softmax(dim=0)
    top_k_ids = torch.topk(mask_token_logits, k).indices.tolist()
    logits = mask_token_logits[top_k_ids]
    probs = mask_token_probs[top_k_ids]
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids)
    return probs, top_k_tokens

def get_idiom_subs(tokenizer, model, source_sent, mask_word, max_length):
    mask_sentence = source_sent.replace(mask_word, '[MASK]'*4)
    probs, top_5_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, 5)
    for _ in range(3):
        for i in range(5):
            temp_sentence = mask_sentence.replace('[MASK]', top_5_tokens[i], 1)
            probs_sub, top_token = predict_char(tokenizer, model, source_sent, temp_sentence, max_length, 1)
            probs[i] *= probs_sub[0]
            top_5_tokens[i] += top_token[0]
    return probs, top_5_tokens

def get_word_subs(tokenizer, model, source_sent, mask_word, max_length):
    mask_sentence = source_sent.replace(mask_word, '[MASK]'*2)
    probs_first, top_5_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, 5)
    substitution_words = []
    word_probs = []
    for i in range(5):
        temp_sentence = mask_sentence.replace('[MASK]', top_5_tokens[i], 1)
        probs_second, top_3_tokens = predict_char(tokenizer, model, source_sent, temp_sentence, max_length, 3)
        probs = [(probs_first[i] * next_prob).item() for next_prob in probs_second]
        words = [top_5_tokens[i] + next_char for next_char in top_3_tokens]
        word_probs += probs
        substitution_words += words

    return word_probs, substitution_words

def get_char_subs(tokenizer, model, source_sent, mask_char, max_length, k):
    mask_sentence = source_sent.replace(mask_char, '[MASK]')
    probs, top_k_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, k)
    probs = [prob.item() for prob in probs]
    return probs, top_k_tokens

def save_results(results, output_path):
    with open(output_path, 'w+', encoding='utf-8') as f_result:
        for r in results:
            f_result.write(' '.join(r) + '\n')

def generate(model, tokenizer, eval_file, max_length):
    
    sentences, mask_words = read_eval_dataset(eval_file)

    model.to('cuda')
    model.eval()

    results = []

    for i in range(len(sentences)):
        len_word = len(mask_words[i])
        if len(sentences[i]) > int((max_length-3) / 2):
            start_index = sentences[i].index(mask_words[i])
            end_index = start_index + len_word
            sentences[i] = truncate(sentences[i], start_index, end_index, int((max_length-3) / 2))
        len_sentence = len(sentences[i])
        probs, substitutions = get_word_subs(tokenizer, model, sentences[i], mask_words[i], max_length)
        if len_word == 4:
            _, idiom_substitutions = get_idiom_subs(tokenizer, model, sentences[i], mask_words[i], max_length)
            substitutions.extend(idiom_substitutions)
        if len_word == 2:
            _, one_char_word_substitutions = get_char_subs(tokenizer, model, sentences[i], mask_words[i], max_length, 2)
            substitutions.extend(one_char_word_substitutions)
        if len_word == 1:
            _, one_char_word_substitutions = get_char_subs(tokenizer, model, sentences[i], mask_words[i], max_length, 3)
            substitutions.extend(one_char_word_substitutions)
        results.append(substitutions)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default=None, type=str,
                        help="BERT pre-trained model name")
    parser.add_argument("--model_cache", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--eval_file", default=None, type=str, required=True,
                        help="The path to the dataset")
    parser.add_argument("--output_path", default=None, type=str, required=True,
                        help="The output path where the model predictions will be written")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                            "Sequences longer than this will be truncated, and sequences shorter \n"
                            "than this will be padded.")

    args = parser.parse_args()

    model_name = args.bert_model
    model_cache = args.model_cache
    eval_file = args.eval_file
    output_path = args.output_path
    max_length = args.max_seq_length

    logger.info("loading model...")

    if model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_cache)
        model = AutoModelForMaskedLM.from_pretrained(model_cache)
    elif model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    else:
        raise(ValueError("At least one of bert_model and model_cache should be given"))
    
    print(model)
    print(tokenizer)

    results = generate(model, tokenizer, eval_file, max_length)
    save_results(results, output_path)

if __name__ == '__main__':
    main()