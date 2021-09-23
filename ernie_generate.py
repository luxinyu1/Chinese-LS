import numpy as np
import sys
from sklearn.metrics import f1_score
import paddle as P
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D
from collections import namedtuple

from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModelForPretraining, ErnieModel

class ErnieGenerate(ErnieModelForPretraining):
    def __init__(self, *args, **kwargs):
        super(ErnieGenerate, self).__init__(*args, **kwargs)
        del self.pooler_heads
    def forward(self, src_ids, *args, **kwargs):
        pooled, encoded = ErnieModel.forward(self, src_ids, *args, **kwargs)
        encoded_2d = L.gather_nd(encoded, L.where(src_ids == mask_id))
        encoded_2d = self.mlm(encoded_2d)
        encoded_2d = self.mlm_ln(encoded_2d)
        logits_2d = L.matmul(encoded_2d, self.word_emb.weight, transpose_y=True) + self.mlm_bias
        return logits_2d

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

def pre_process(sentence, difficult_word, mask_num):
    return sentence.replace(difficult_word, '[MASK]'*mask_num)

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, value = line.strip().split(',')
            dict[key] = value
    return dict

def save_results(results, output_path):
    with open(output_path, 'a', encoding='utf-8') as f_result:
        f_result.write(' '.join(results) + '\n')

EVAL_PATH = './dataset/annotation_data.csv'
MODEL_DIR = './model/ernie1.0.1'
OUTPUT_PATH = './data/ernie_output.csv'
SUBSITUTION_NUM = 10

eval_path = EVAL_PATH
model_dir = MODEL_DIR
substitution_num = SUBSITUTION_NUM
output_path = OUTPUT_PATH

sentences, difficult_words = read_dataset(eval_path)

place = F.CUDAPlace(D.parallel.Env().dev_id)
D.guard(place).__enter__()

# 初始化tokenizer
tokenizer = ErnieTokenizer.from_pretrained(model_dir)
rev_dict = {v: k for k, v in tokenizer.vocab.items()}
rev_dict[tokenizer.pad_id] = '' # replace [PAD]
rev_dict[tokenizer.unk_id] = '' # replace [PAD]

@np.vectorize
def rev_lookup(i):
    return rev_dict[i]

ernie = ErnieGenerate.from_pretrained(model_dir)

for sentence, difficult_word in zip(sentences, difficult_words):
    print(sentence, difficult_word)
    # 词预测
    ids, _ = tokenizer.encode(sentence, pre_process(sentence, difficult_word, 2))
    # print(ids)
    src_ids = D.to_variable(np.expand_dims(ids, 0))
    mask_id = tokenizer.mask_id
    mask_index = np.argwhere(ids==mask_id)[0]
    logits = ernie(src_ids)
    _, top_5_tokens = L.topk(logits, 5)
    # print(top_k_tokens[1].numpy())
    substitution_words = []
    for token in top_5_tokens[0].numpy():
        first_char = str(rev_lookup(token))
        ids[mask_index] = token
        # sep_index = np.argwhere(ids==tokenizer.sep_id)[0][0]
        # second_ids = ids[sep_index::]
        # second_ids[0:0] = tokenizer.cls_id
        second_ids = D.to_variable(np.expand_dims(ids, 0))
        logits = ernie(second_ids)
        _, top_3_tokens = L.topk(logits, 3)
        for _token in top_3_tokens[0].numpy():
            second_char = str(rev_lookup(_token))
            substitution_words.append(first_char+second_char)
    # 字预测
    if len(difficult_word) == 1 or len(difficult_word) == 2:
        ids, _ = tokenizer.encode(sentence, pre_process(sentence, difficult_word, 1))
        src_ids = D.to_variable(np.expand_dims(ids, 0))
        mask_id = tokenizer.mask_id
        mask_index = np.argwhere(ids==mask_id)[0]
        logits = ernie(src_ids)
        _, top_3_tokens = L.topk(logits, 3)
        chars = [str(rev_lookup(token)) for token in top_3_tokens[0].numpy()]
        substitution_words.extend(chars)
    # 成语预测
    if len(difficult_word) == 4:
        ids, _ = tokenizer.encode(sentence, pre_process(sentence, difficult_word, 4))
        mask_id = tokenizer.mask_id
        src_ids = D.to_variable(np.expand_dims(ids, 0))
        logits = ernie(src_ids)
        _, top_5_tokens = L.topk(logits, 5)
        # decoded = rev_lookup(top_5_tokens.numpy())
        # print(decoded)
        # sep_index = np.argwhere(ids==tokenizer.sep_id)[0][0]
        # ids = ids[sep_index::]
        # ids[0:0] = tokenizer.cls_id
        for token in top_5_tokens[0].numpy():
            second_ids = ids.copy()
            first_char = str(rev_lookup(token))
            mask_index = np.argwhere(second_ids==mask_id)[0][0]
            second_ids[mask_index] = token
            src_ids = D.to_variable(np.expand_dims(second_ids, 0))
            logits = ernie(src_ids).numpy()
            top_token = np.argmax(logits[0], -1)
            second_char = str(rev_lookup(top_token))
            mask_index = np.argwhere(second_ids==mask_id)[0][0]
            second_ids[mask_index] = top_token
            src_ids = D.to_variable(np.expand_dims(second_ids, 0))
            logits = ernie(src_ids).numpy()
            top_token = np.argmax(logits[0], -1)
            third_char = str(rev_lookup(top_token))
            mask_index = np.argwhere(second_ids==mask_id)[0][0]
            second_ids[mask_index] = top_token
            src_ids = D.to_variable(np.expand_dims(second_ids, 0))
            logits = ernie(src_ids).numpy()
            top_token = np.argmax(logits, -1)
            forth_char = str(rev_lookup(top_token[0]))
            substitution_words.append(first_char+second_char+third_char+forth_char)
    save_results(substitution_words, output_path)