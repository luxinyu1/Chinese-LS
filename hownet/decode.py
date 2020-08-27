import pickle

with open('./hownet/dict.pkl', 'rb') as fh:
    dict = pickle.load(fh)

with open('./hownet/word_candidates_sense.pkl','rb') as fp:
    word_candidate = pickle.load(fp)

word_candidates_decoded = {}

for index, dic in word_candidate.items():
    orig_word = dict.get(index)
    sub_dic_decoded = {}
    for ch, codes in dic.items():
        sub_dic_decoded[ch] = [dict.get(code) for code in codes]
    word_candidates_decoded[orig_word] = sub_dic_decoded

with open('./hownet/word_candidates_decoded.pkl', 'wb') as f:
    pickle.dump(word_candidates_decoded, f)