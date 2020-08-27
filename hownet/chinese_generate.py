import pickle
import OpenHowNet

from tqdm import tqdm

hownet_dict = OpenHowNet.HowNetDict()

word_candidate = {}

dict = {}
zh_word_list = hownet_dict.get_zh_words()
zh_word_list = [word for word in zh_word_list if len(word) in range(2,5)]

for code, word in enumerate(zh_word_list):
    dict[code] = word

f = open('./hownet/dict.pkl', 'wb')
pickle.dump(dict, f)

word_pos = {}
word_sem = {}

pos_list = ['noun', 'verb', 'adj', 'adv']
pos_set = set(pos_list)

for i1, w1 in dict.items():
    # print(i1, w1)
    try:
        tree = hownet_dict.get_sememes_by_word(w1, merge=False, structured=True, lang="zh")
        w1_sememes = hownet_dict.get_sememes_by_word(w1, structured=False, lang="zh", merge=False)
        new_w1_sememes = [t['sememes'] for t in w1_sememes]

        w1_pos_list = [x['word']['ch_grammar'] for x in tree]
        word_pos[i1] = w1_pos_list
        word_sem[i1] = new_w1_sememes
        
    except:
        word_pos[i1] = []
        word_sem[i1] = []

print('POS and sememe generated.')

def add_w1(w1, i1):
    # 初始化i1的候选词
    word_candidate[i1] = {}
    w1_s_flag = 0

    # 把这个词所有不同的词性全取出来
    w1_pos = set(word_pos[i1])
    # 给每个词性初始化一个候选词表
    for pos in pos_set:
        word_candidate[i1][pos] = []
    # 将刚取出来的词性与pos_set = {'noun', 'verb', 'adj', 'adv'}算交集
    valid_pos_w1 = w1_pos & pos_set

    # 如果交集为空，那这个词就不生成候选词
    if len(valid_pos_w1) == 0:
        return

    # 把这个词所有id的扁平化的义元树取出来
    new_w1_sememes = word_sem[i1]

    # 一个义元树也没有，返回
    if len(new_w1_sememes) == 0:
        return

    # 遍历vacab.txt里的所有索引，词
    for i2, w2 in dict.items():
        # 计算词和目标词一样，跳过
        if i1 == i2:
            continue
        # 同样的，也找一遍所有词性
        w2_pos = set(word_pos[i2])
        all_pos = w2_pos & w1_pos & pos_set
        # 是否它和w1词性有交集且和pos_set = {'noun', 'verb', 'adj', 'adv'}有交集，没有的话跳过
        if len(all_pos) == 0:
            continue
        # 找所有义元树
        new_w2_sememes = word_sem[i2]
        # print(w1)
        # print(w2)
        # print('w1:\n', new_w1_sememes)
        # print('w2:\n', new_w2_sememes)
        # 没有，也跳过
        if len(new_w2_sememes) == 0:
            continue
        # not_in_num1 = count(w1_sememes, w2_sememes)
        # not_in_num2 = count(w2_sememes,w1_sememes)
        # not_in_num=not_in_num1+not_in_num2
        w_flag = 0

        # 遍历目标词所有义元树
        for s1_id in range(len(new_w1_sememes)):
            if w_flag == 1:
                break
            # 义元树对应的词性
            pos_w1 = word_pos[i1][s1_id]
            # 义元树s1
            s1 = set(new_w1_sememes[s1_id])
            # 词性不在pos_set = {'noun', 'verb', 'adj', 'adv'}里，跳过
            if pos_w1 not in pos_set:
                continue
            for s2_id in range(len(new_w2_sememes)):
                if w_flag==1:
                    break
                pos_w2 = word_pos[i2][s2_id]
                s2 = set(new_w2_sememes[s2_id])
                if pos_w1 == pos_w2 and s1 == s2:
                    word_candidate[i1][pos_w1].append(i2)
                    w_flag = 1
                    break

with tqdm(total=len(dict.items())) as t1:
    for i1, w1 in dict.items():
        t1.update(1)
        add_w1(w1, i1)

f = open('./hownet/word_candidates_sense.pkl', 'wb')
pickle.dump(word_candidate, f)