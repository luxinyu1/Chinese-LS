def read_dict(dict_path):
    dict = []
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            entry = line[9:].strip().split(' ')
            if entry:
                dict.append(entry)
    return dict

def read_eval_dataset(data_path):
    sentences = []
    difficult_words = []
    with open(data_path, 'r', encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            row = line.strip().split('\t')
            sentence, difficult_word = row[0], row[1]
            sentences.append(''.join(sentence.split(' ')))
            difficult_words.append(difficult_word)
    return sentences, difficult_words

def save_results(result, output_path):
    with open(output_path, 'a', encoding='utf-8') as f_result:
        f_result.write(' '.join(result) + '\n')

def main():
    DICT_PATH = './dict/HIT-dict=.txt'
    DATA_PATH = './dataset/annotation_data.csv'
    OUTPUT_PATH = './data/dict_output.csv'
    
    dict_path = DICT_PATH
    data_path = DATA_PATH
    output_path = OUTPUT_PATH

    dict = read_dict(dict_path)

    sentences, difficult_words = read_eval_dataset(data_path)

    substitution_words = []
    
    for difficult_word in difficult_words:
        isFound = False
        substitution_words = []
        for entry in dict:
            if difficult_word in entry:
                isFound = True
                for word in entry:
                    substitution_words.append(word)
        if (isFound == False):
            substitution_words.append('NULL')

        print(substitution_words)
        save_results(substitution_words, output_path)

if __name__ == '__main__':
    main()