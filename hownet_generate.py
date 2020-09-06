import pickle

def read_dataset(eval_path):
    difficult_words = []
    pos_tags = []
    with open(eval_path, 'r', encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            row = line.strip().split('\t')
            difficult_word, pos_tag = row[1], row[2]
            difficult_words.append(difficult_word)
            pos_tags.append(pos_tag)
    return difficult_words, pos_tags

def save_res(res, output_path):
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(' '.join(res) + '\n')

def main():
    EVAL_PATH = './dataset/annotation_data.csv'
    OUTPUT_PATH = './data/hownet_output.csv'

    eval_path = EVAL_PATH
    output_path = OUTPUT_PATH

    difficult_words, pos_tags = read_dataset(eval_path)

    with open('./hownet/word_candidates_decoded.pkl','rb') as fp:
        word_candidates = pickle.load(fp)
    
    for difficult_word, pos_tag in zip(difficult_words, pos_tags):
        try:
            res = word_candidates[difficult_word].get('noun')
            res.extend(word_candidates[difficult_word].get('verb'))
            res.extend(word_candidates[difficult_word].get('adj'))
            res.extend(word_candidates[difficult_word].get('adv'))
            res = [word for word in res if len(word) <= len(difficult_word)]
        except:
            res = ['NULL']
        if len(res)==0:
            res.append('NULL')
        save_res(res, output_path)

if __name__ == '__main__':
    main()