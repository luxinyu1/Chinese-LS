import gensim

def read_freq_dict(freq_dict_path):
    freq_dict = {}
    with open(freq_dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, value = line.strip().split(',')
            freq_dict[key] = value
    return freq_dict

def read_dataset(path):
    sentences = []
    difficult_words = []
    mask_indexs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split('\t')
            sentences.append(''.join(row[0].split(' ')))
            difficult_words.append(row[1])
    return sentences, difficult_words

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, value = line.strip().split(',')
            dict[key] = value
    return dict

def save_results(sim_words, output_path):
    with open(output_path, 'a', encoding='utf-8') as f_result:
        f_result.write(' '.join(sim_words) + '\n')

def main():
    EVAL_FILE = './dataset/annotation_data.csv'
    WORD_FREQ_DICT = './data/word_counted.csv'
    OUTPUT_PATH = './data/vector_output.csv'
    WORD_2_VECTOR_MODEL_DIR = './model/merge_sgns_bigram_char300.txt'

    eval_file = EVAL_FILE
    freq_dict_path = WORD_FREQ_DICT
    output_path = OUTPUT_PATH
    word_2_vector_model_dir = WORD_2_VECTOR_MODEL_DIR

    sentences, difficult_words = read_dataset(eval_file)
    freq_dict = read_freq_dict(freq_dict_path)

    for difficult_word in difficult_words:
        sim_words = []
        model_word2vector = gensim.models.KeyedVectors.load_word2vec_format(word_2_vector_model_dir, binary=False)
        try:
            sim_words = model_word2vector.most_similar(difficult_word)
            sim_words = [item[0] for item in sim_words]
        except:
            sim_words.append('NULL')
        print(sim_words)
        save_results(sim_words, output_path)

if __name__ == '__main__':
    main()