import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import jieba
import gensim
from transformers import BertTokenizer, BertForMaskedLM
from scipy.special import softmax
import numpy as np
import traceback
import OpenHowNet

def substitute_ranking(row_line, model_word2vector, model, tokenizer, hownet, source_sentence, source_word, substitution_words, word_freq_dict, substitution_num):

    MAX = 56065
    
    loss_scores = []
    freq_scores = []
    sim_scores = []
    hownet_scores = []
    length_scores = []

    for i in range(len(substitution_words)):
        word = substitution_words[i]
        try:
            freq_scores.append(int(word_freq_dict[word]))
        except:
            freq_scores.append(0)
        sentence_splited = row_line.split('\t')[0].split(' ')
        loss = sent_loss(model, tokenizer, source_sentence, source_word, substitution_words[i])
        loss_scores.append(loss)
        try:
            similarity = model_word2vector.similarity(source_word, word)
            sim_scores.append(similarity)
        except:
            sim_scores.append(0)
        try:
            similarity = hownet.calculate_word_similarity(source_word, word)
            hownet_scores.append(similarity)
        except:
            hownet_scores.append(0)

        length_scores.append(len(word))

    print(substitution_words)
    assert len(loss_scores) == len(freq_scores) == len(sim_scores) == len(hownet_scores) == len(length_scores)
    loss_scores_sorted = sorted(loss_scores)
    loss_ranks = [loss_scores_sorted.index(x) + 1 for x in loss_scores]
    freq_scores_sorted = sorted(freq_scores)
    freq_ranks = [freq_scores_sorted.index(x) + 1 for x in freq_scores]
    sim_scores_sorted = sorted(sim_scores, reverse=True)
    sim_ranks = [sim_scores_sorted.index(x) + 1 for x in sim_scores]
    hownet_scores_sorted = sorted(hownet_scores, reverse=True)
    hownet_ranks = [hownet_scores_sorted.index(x) + 1 for x in hownet_scores]
    length_scores_sorted = sorted(length_scores)
    length_ranks = [length_scores_sorted.index(x) + 1 for x in length_scores]
    all_ranks = [[substitution_word, loss+freq+sim+hownet+length] for substitution_word, loss, freq, sim, hownet, length in zip(substitution_words, loss_ranks, freq_ranks, sim_ranks, hownet_ranks, length_ranks)]
    ss_sorted = sorted(all_ranks, key=lambda x:x[1])
    ss_sorted = [x[0] for x in ss_sorted]
    freq_rank_source = int(word_freq_dict[source_word]) if source_word in word_freq_dict else MAX
    try:
        freq_rank_next = int(word_freq_dict[ss_sorted[1]])
    except:
        freq_rank_next = MAX - 1
    if ss_sorted[0] == source_word and freq_rank_source > freq_rank_next and len(ss_sorted)>=2:
        pre_word = ss_sorted[1]
    else:
        pre_word = ss_sorted[0]
    print(pre_word, ss_sorted)

    return pre_word, ss_sorted[:substitution_num:]

def cut_out(sentence_splited, difficult_word, radius):
    d_index = sentence_splited.index(difficult_word)
    start_index = d_index - radius if d_index - radius > 0 else 0
    end_index = d_index + radius if d_index + radius < len(sentence_splited) else len(sentence_splited) - 1
    sentence = ''.join(sentence_splited[start_index:end_index:])
    return sentence

def cross_entropy_word(X, i, pos):
    X = softmax(X, axis=1)
    loss = 0
    loss -= np.log10(X[i, pos])
    return loss

def sent_loss(model, tokenizer, source_sentence, source_word, substitution_word):
    masked_sentence = source_sentence.replace(source_word, '[MASK]'*len(source_word))
    label_sentence = source_sentence.replace(source_word, substitution_word)
    print(masked_sentence)
    input_ids = tokenizer.encode(masked_sentence, return_tensors='pt')
    label_ids = tokenizer.encode(label_sentence, return_tensors='pt')
    input_ids = input_ids.to('cuda')
    label_ids = label_ids.to('cuda')
    with torch.no_grad():
        outputs = model(input_ids, masked_lm_labels=label_ids)
    loss, prediction_scores = outputs[:2]
    print(loss)

    return loss

def read_ss_result(res_path):
    res = []
    with open(res_path, 'r', encoding='utf-8') as f_res:
        for line in f_res:
            res.append(line.strip().split(' '))
    return res

def read_dataset(data_path):
    sentences = []
    words = []
    row_lines = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            row_lines.append(line)
            if not line:
                break
            row = line.strip().split('\t')
            sentence, word = row[0], row[1]
            sentences.append(''.join(sentence.split(' ')))
            words.append(word)
    return row_lines, sentences, words

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, pingyin, value = line.strip().split('\t')
            dict[key] = value
    return dict

def save_result(row_line, pre_word, ss_sorted, path):
    with open(path, 'a', encoding='utf-8') as f_ss_res:
        f_ss_res.write(row_line.strip() + '\n' + pre_word + '\n' + ' '.join(ss_sorted) + '\n')

def main():

    MODEL_CACHE = './model/bert-base-chinese'
    WORD_2_VECTOR_MODEL_DIR = './model/merge_sgns_bigram_char300.txt'

    WORD_FREQ_DICT = './dict/modern_chinese_word_freq.txt'

    EVAL_FILE_PATH = './dataset/annotation_data.csv'
    BERT_RES_PATH = './data/bert_ss_res.csv'
    # ERNIE_RES_PATH = './data/ernie_output.csv'
    VECTOR_RES_PATH = './data/vector_ss_res.csv'
    DICT_RES_PATH = './data/dict_ss_res.csv'
    HOWNET_RES_PATH = './data/hownet_ss_res.csv'
    MIX_RES_PATH = './data/mix_ss_res.csv'

    SUBSTITUTION_NUM = 10

    word_2_vector_model_dir = WORD_2_VECTOR_MODEL_DIR
    model_cache = MODEL_CACHE

    word_freq_dict = WORD_FREQ_DICT

    eval_file_path = EVAL_FILE_PATH

    bert_res_path = BERT_RES_PATH
    # ernie_res_path = ERNIE_RES_PATH
    vector_res_path = VECTOR_RES_PATH
    dict_res_path = DICT_RES_PATH
    hownet_res_path = HOWNET_RES_PATH
    mix_res_path = MIX_RES_PATH

    substitution_num = SUBSTITUTION_NUM

    print('loading models...')
    tokenizer = BertTokenizer.from_pretrained(model_cache)
    model = BertForMaskedLM.from_pretrained(model_cache)
    # OpenHowNet.download()
    hownet = OpenHowNet.HowNetDict(use_sim=True)
    model.to('cuda')
    model.eval()
    print('loading embeddings...')
    model_word2vector = gensim.models.KeyedVectors.load_word2vec_format(word_2_vector_model_dir, binary=False)
    print('loading files...')
    word_freq_dict = read_dict(word_freq_dict)

    bert_res = read_ss_result(bert_res_path)
    vector_res = read_ss_result(vector_res_path)
    dict_res = read_ss_result(dict_res_path)
    hownet_res = read_ss_result(hownet_res_path)
    mix_res = read_ss_result(mix_res_path)

    row_lines, source_sentences, source_words = read_dataset(eval_file_path)

    for row_line, source_sentence, source_word, bert_subs, vector_subs, dict_subs, hownet_subs, mix_subs in zip(row_lines, source_sentences, source_words, bert_res, vector_res, dict_res, hownet_res, mix_res):
        # 全部运行可能耗时较长，建议注释部分代码块运行需要的测试
        if bert_subs[0] != 'NULL':
            bert_pre_word, bert_ss_sorted = substitute_ranking(row_line, model_word2vector, model, tokenizer, hownet, source_sentence, source_word, bert_subs, word_freq_dict, substitution_num)
        else:
            bert_pre_word = 'NULL'
            bert_ss_sorted = ['NULL']
        # if vector_subs[0] != 'NULL':
        #     vector_pre_word, vector_ss_sorted = substitute_ranking(row_line, model_word2vector, model, tokenizer, hownet, source_sentence, source_word, vector_subs, word_freq_dict, substitution_num)
        # else:
        #     vector_pre_word = 'NULL'
        #     vector_ss_sorted = ['NULL']
        # if dict_subs[0] != 'NULL':
        #     dict_pre_word, dict_ss_sorted = substitute_ranking(row_line, model_word2vector, model, tokenizer, hownet, source_sentence, source_word, dict_subs, word_freq_dict, substitution_num)
        # else:
        #     dict_pre_word = 'NULL'
        #     dict_ss_sorted = ['NULL']
        # if hownet_subs[0] != 'NULL':
        #     hownet_pre_word, hownet_ss_sorted = substitute_ranking(row_line, model_word2vector, model, tokenizer, hownet, source_sentence, source_word, hownet_subs, word_freq_dict, substitution_num)
        # else:
        #     hownet_pre_word = 'NULL'
        #     hownet_ss_sorted = ['NULL']
        # if mix_subs[0] != 'NULL':
        #     mix_pre_word, mix_ss_sorted = substitute_ranking(row_line, model_word2vector, model, tokenizer, hownet, source_sentence, source_word, mix_subs, word_freq_dict, substitution_num)
        # else:
        #     mix_pre_word = 'NULL'
        #     mix_ss_sorted = ['NULL']

        # save_result(row_line, bert_pre_word, bert_ss_sorted, './data/bert_sr_res.csv')
        # save_result(row_line, ernie_pre_word, ernie_ss_sorted, './data/ernie_sr_res.csv')
        # save_result(row_line, vector_pre_word, vector_ss_sorted, './data/vector_sr_res.csv')
        save_result(row_line, dict_pre_word, dict_ss_sorted, './data/dict_sr_res.csv')
        save_result(row_line, hownet_pre_word, hownet_ss_sorted, './data/hownet_sr_res.csv')
        save_result(row_line, mix_pre_word, mix_ss_sorted, './data/mix_sr_res.csv')

if __name__ == '__main__':
    main()