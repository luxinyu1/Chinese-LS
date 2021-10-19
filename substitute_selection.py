import jieba

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, pingyin, value = line.strip().split('\t')
            dict[key] = value
    return dict

def read_generate_result(res_path):
    res = []
    with open(res_path, 'r', encoding='utf-8') as f_res:
        for line in f_res:
            res.append(line.strip().split(' '))
    return res

def substitute_selection(gen_res, word_list, origin_words):
    valid_res = []
    for origin_word, item in zip(origin_words, gen_res):
        valid_words = []
        if item[0] == 'NULL':
            valid_res.append(['NULL'])
            continue
        for res in item:
            subs_all_in = False
            if len(res) > 2:
                subs_all_in = True
                subs = jieba.lcut(res)
                for sub in subs:
                    if sub not in word_list:
                        subs_all_in = False
            if (res in word_list or subs_all_in) and res not in valid_words:
                valid_words.append(res)
        if valid_words:
            valid_res.append(valid_words)
        else:
            valid_res.append([origin_word])
    return valid_res

def read_dataset(data_path):
    words = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            row = line.strip().split('\t')
            word = row[1]
            words.append(word)
    return words

def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_result:
        for item in results:
            f_result.write(' '.join(item) + '\n')

DATASET_PATH = './dataset/annotation_data.csv'

BERT_RES_PATH = './data/bert_output.csv'
BERT_NO_AUTOREGRESSIVE_RES_PATH = './data/bert_no_autoregressive_output.csv'
ERNIE_RES_PATH = './data/ernie_output.csv'
BERT_WWM_RES_PATH = './data/bert_wwm_output.csv'
BERT_WWM_EXT_RES_PATH = './data/bert_wwm_ext_output.csv'
ROBERTA_WWM_EXT_RES_PATH = './data/roberta_wwm_ext_output.csv'
MACBERT_RES_PATH = './data/macbert_output.csv'
ELECTRA_RES_PATH = './data/electra_output.csv'
VECTOR_RES_PATH = './data/vector_output.csv'
DICT_RES_PATH = './data/dict_output.csv'
HOWNET_RES_PATH = './data/hownet_output.csv'
HYBRID_RES_PATH = './data/hybrid_output.csv'

WORD_FREQ_DICT = './dict/modern_chinese_word_freq.txt'

dataset_path = DATASET_PATH

bert_res_path = BERT_RES_PATH
bert_no_autoregressive_res_path = BERT_NO_AUTOREGRESSIVE_RES_PATH
ernie_res_path = ERNIE_RES_PATH
bert_wwm_res_path = BERT_WWM_RES_PATH
bert_wwm_ext_res_path = BERT_WWM_EXT_RES_PATH
roberta_wwm_ext_res_path = ROBERTA_WWM_EXT_RES_PATH
macbert_res_path = MACBERT_RES_PATH
electra_res_path = ELECTRA_RES_PATH
vector_res_path = VECTOR_RES_PATH
dict_res_path = DICT_RES_PATH
hownet_res_path = HOWNET_RES_PATH
hybrid_res_path = HYBRID_RES_PATH

word_freq_dict = WORD_FREQ_DICT

word_list = read_dict(word_freq_dict)
origin_words = read_dataset(dataset_path)

bert_res = read_generate_result(bert_res_path)
bert_no_autoregressive_res = read_generate_result(bert_no_autoregressive_res_path)
ernie_res = read_generate_result(ernie_res_path)
bert_wwm_res = read_generate_result(bert_wwm_res_path)
bert_wwm_ext_res = read_generate_result(bert_wwm_ext_res_path)
roberta_wwm_ext_res = read_generate_result(roberta_wwm_ext_res_path)
macbert_res = read_generate_result(macbert_res_path)
electra_res = read_generate_result(electra_res_path)
vector_res = read_generate_result(vector_res_path)
dict_res = read_generate_result(dict_res_path)
hownet_res = read_generate_result(hownet_res_path)
hybrid_res = read_generate_result(hybrid_res_path)

valid_bert_res = substitute_selection(bert_res, word_list, origin_words)
valid_bert_no_autoregressive_res = substitute_selection(bert_no_autoregressive_res, word_list, origin_words)
valid_bert_wwm_res = substitute_selection(bert_wwm_res, word_list, origin_words)
valid_bert_wwm_ext_res = substitute_selection(bert_wwm_ext_res, word_list, origin_words)
valid_ernie_res = substitute_selection(ernie_res, word_list, origin_words)
valid_roberta_wwm_ext_res = substitute_selection(roberta_wwm_ext_res, word_list, origin_words)
valid_macbert_res = substitute_selection(macbert_res, word_list, origin_words)
valid_electra_res = substitute_selection(electra_res, word_list, origin_words)
valid_vector_res = substitute_selection(vector_res, word_list, origin_words)
valid_dict_res = substitute_selection(dict_res, word_list, origin_words)
valid_hownet_res = substitute_selection(hownet_res, word_list, origin_words) 
valid_hybrid_res = substitute_selection(hybrid_res, word_list, origin_words)

save_results(valid_bert_res, './data/bert_ss_res.csv')
save_results(valid_bert_no_autoregressive_res, './data/bert_no_autoregressive_ss_res.csv')
save_results(valid_bert_wwm_res, './data/bert_wwm_ss_res.csv')
save_results(valid_ernie_res, './data/ernie_ss_res.csv')
save_results(valid_bert_wwm_ext_res, './data/bert_wwm_ext_ss_res.csv')
save_results(valid_roberta_wwm_ext_res, './data/roberta_wwm_ext_ss_res.csv')
save_results(valid_macbert_res, './data/macbert_base_ss_res.csv')
save_results(valid_electra_res, './data/electra_ss_res.csv')
save_results(valid_vector_res, './data/vector_ss_res.csv')
save_results(valid_dict_res, './data/dict_ss_res.csv')
save_results(valid_hownet_res, './data/hownet_ss_res.csv')
save_results(valid_hybrid_res, './data/hybrid_ss_res.csv')