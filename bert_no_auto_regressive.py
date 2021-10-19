import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM

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
    sequence_dict = encoder(tokenizer, source_sent, mask_sentence, max_length)
    input_ids = sequence_dict['input_ids'].to('cuda')
    attention_masks = sequence_dict['attention_mask'].to('cuda')
    token_type_ids = sequence_dict['token_type_ids'].to('cuda')
    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(input_ids, attention_masks, token_type_ids) # Return type: tuple(torch.FloatTensor) comprising various elements depending on the configuration (BertConfig) and inputs
    logits = outputs[0]
    mask_tokens_logits = logits[0, masked_index, :]
    mask_tokens_probs = mask_tokens_logits.softmax(dim=1)
    k = 5
    top_k_probs, top_k_ids = torch.topk(mask_tokens_probs, k)
    top_k_probs = top_k_probs.tolist()
    top_k_tokens = []
    for i in range(4):
        top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k_ids[i]))
    final_subs = []
    for (first_token, first_prob) in zip(top_k_tokens[0], top_k_probs[0]):
        for (second_token, second_prob) in zip(top_k_tokens[1], top_k_probs[1]):
            for (third_token, third_prob) in zip(top_k_tokens[2], top_k_probs[2]):
                for (forth_token, forth_prob) in zip(top_k_tokens[3], top_k_probs[3]):
                    word = first_token + second_token + third_token + forth_token
                    if word not in [s[0] for s in final_subs]:
                        final_subs.append((first_token+second_token+third_token+forth_token, first_prob*second_prob*third_prob*forth_prob))

    return [x[1] for x in final_subs], [x[0] for x in final_subs]

def get_word_subs(tokenizer, model, source_sent, mask_word, max_length):
    mask_sentence = source_sent.replace(mask_word, '[MASK]'*2)
    sequence_dict = encoder(tokenizer, source_sent, mask_sentence, max_length)
    input_ids = sequence_dict['input_ids'].to('cuda')
    attention_masks = sequence_dict['attention_mask'].to('cuda')
    token_type_ids = sequence_dict['token_type_ids'].to('cuda')
    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    with torch.no_grad():
        outputs = model(input_ids, attention_masks, token_type_ids) # Return type: tuple(torch.FloatTensor) comprising various elements depending on the configuration (BertConfig) and inputs
    logits = outputs[0]
    mask_tokens_logits = logits[0, masked_index, :]
    mask_tokens_probs = mask_tokens_logits.softmax(dim=1)
    k = 5
    top_k_probs, top_k_ids = torch.topk(mask_tokens_probs, k)
    top_k_probs = top_k_probs.tolist()
    top_k_tokens = []
    top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k_ids[0]))
    top_k_tokens.append(tokenizer.convert_ids_to_tokens(top_k_ids[1]))
    final_subs = []
    for (first_token, first_prob) in zip(top_k_tokens[0], top_k_probs[0]):
        for (second_token, second_prob) in zip(top_k_tokens[1], top_k_probs[1]):
            word = first_token + second_token
            if word not in [s[0] for s in final_subs]:
                final_subs.append((first_token+second_token, first_prob*second_prob))

    return [x[1] for x in final_subs], [x[0] for x in final_subs]

def get_char_subs(tokenizer, model, source_sent, mask_char, max_length, k):
    mask_sentence = source_sent.replace(mask_char, '[MASK]')
    logits, top_k_tokens = predict_char(tokenizer, model, source_sent, mask_sentence, max_length, k)
    logits = logits.tolist()
    return logits, top_k_tokens

def save_results(results, output_path):
    with open(output_path, 'a', encoding='utf-8') as f_result:
        f_result.write(' '.join(results) + '\n')

def main():
    # file-path
    MODEL_CACHE = './model/bert-base-chinese'
    EVAL_FILE = './dataset/annotation_data.csv'
    OUTPUT_PATH = './data/bert_no_autoregressive_output.csv'
    # hyperparameter
    MAX_LENGH = 128
    SUBSITUTION_NUM = 10

    model_cache = MODEL_CACHE
    eval_file = EVAL_FILE
    output_path = OUTPUT_PATH
    max_length = MAX_LENGH
    substitution_num = SUBSITUTION_NUM

    print('loading model...')
    tokenizer = BertTokenizer.from_pretrained(model_cache)
    model = BertForMaskedLM.from_pretrained(model_cache)
    
    sentences, mask_words = read_eval_dataset(eval_file)

    model.to('cuda')
    model.eval()

    results = []

    for i in range(len(sentences)):
        print(sentences[i], mask_words[i])
        len_word = len(mask_words[i])
        if len(sentences[i]) > int((max_length-3) / 2):
            start_index = sentences[i].index(mask_words[i])
            end_index = start_index + len_word
            sentences[i] = truncate(sentences[i], start_index, end_index, int((max_length-3) / 2))
        len_sentence = len(sentences[i])
        logits, substitutions = get_word_subs(tokenizer, model, sentences[i], mask_words[i], max_length)
        if len_word == 4:
            _, idiom_substitutions = get_idiom_subs(tokenizer, model, sentences[i], mask_words[i], max_length)
            substitutions.extend(idiom_substitutions)
        if len_word == 2:
            _, one_char_word_substitutions = get_char_subs(tokenizer, model, sentences[i], mask_words[i], max_length, 2)
            substitutions.extend(one_char_word_substitutions)
        if len_word == 1:
            _, one_char_word_substitutions = get_char_subs(tokenizer, model, sentences[i], mask_words[i], max_length, 3)
            substitutions.extend(one_char_word_substitutions)
        print(substitutions)
        save_results(substitutions, output_path)

if __name__ == '__main__':
    main()