def read_manual_data(path):
    origin_words = []
    manual_labels = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split('\t')
            origin_words.append(row[1])
            manual_labels.append(row[-1])
    
    manual_labels = [x.split(' ') for x in manual_labels]
    assert len(origin_words) == len(manual_labels)
    return origin_words, manual_labels

def read_gen_data(path):
    gen = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            row = line.strip().split(' ')
            gen.append(row)
    return gen

def read_dict(dict_path):
    dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f_dict:
        for line in f_dict:
            key, pingyin, value = line.strip().split('\t')
            dict[key] = value
    return dict

def read_sr_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.readlines()
        content = [line.strip() for line in content]
        pre_words = content[1::3]
        substitution_words = content[2::3]
    substitution_words = [x.split(' ') for x in substitution_words]
    assert len(pre_words) == len(substitution_words)
    return pre_words, substitution_words

def read_eval_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = f.readlines()
        rows = rows[::3]
        rows = [int(label.strip()) for label in rows]
    return rows

def evaluate_dataset(labels, pre_words, origin_words, manual_labels):
    assert len(labels) == len(pre_words) == len(origin_words) == len(manual_labels) 
    unchanged = labels.count(1)
    correct = labels.count(2)
    wrong = labels.count(3)
    ungenerated = labels.count(4)
    changed = correct + wrong
    pre = 0
    cons = 0
    for label, sub, origin, gold in zip(labels, pre_words, origin_words, manual_labels):
        if sub in gold and sub != origin:
            pre += 1
            if label == 2:
                cons += 1
        elif sub not in gold and sub != origin and sub != 'NULL':
            if label == 3:
                cons += 1
    return changed, correct/changed, pre/changed, cons

def evaluate_SS_scores(ss, labels):
    assert len(ss)==len(labels)

    potential = 0
    instances = len(ss)
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0

    for i in range(instances):

        one_prec = 0
        
        common = list(set(ss[i]).intersection(labels[i]))

        if len(common) >= 1:
            potential += 1

        precision += len(common)
        recall += len(common)
        precision_all += len(labels[i])
        recall_all += len(ss[i]) if ss[i][0] != 'NULL' else 0
    
    print("COMMON:", precision)

    potential /= instances
    precision /= precision_all
    recall /= recall_all
    F_score = 2*precision*recall/(precision+recall)

    return potential, precision, recall, F_score

def evaluate_pipeline_scores(substitution_words, source_words, gold_words):

    instances = len(substitution_words)
    precision = 0
    accuracy = 0
    changed_proportion = 0

    for sub, source, gold in zip(substitution_words, source_words, gold_words):
        if sub==source or sub in gold:
            precision += 1
        if sub!=source and sub in gold:
            accuracy += 1
        if sub != source:
            changed_proportion += 1
        # if sub not in gold and sub != source:
        #     print(sub, gold)

    return precision/instances, accuracy/instances, changed_proportion/instances

def evaluate_error(word_freq_rank, manual_labels, pre_words, substitutions):
    assert len(manual_labels) == len(pre_words) == len(substitutions)
    error3a = 0
    error3b = 0
    error4 = 0
    error5 = 0
    noerror = 0
    for i in range(len(substitutions)):
        difficult_labels = [x for x in manual_labels[i] if x not in word_freq_rank or int(word_freq_rank[x]) > 15000]
        simple_labels = list(set(manual_labels[i]) - set(difficult_labels))
        if len(set(manual_labels[i]).intersection(set(substitutions[i]))) == 0:
            error3a += 1
        elif len(set(simple_labels).intersection(set(substitutions[i]))) == 0:
            error3b += 1
        if pre_words[i] not in manual_labels[i]:
            error4 += 1
        elif pre_words[i] not in simple_labels:
            error5 += 1
        else:
            noerror += 1
    return noerror, noerror / len(manual_labels), error3a, error3a / len(manual_labels), error3b, error3b / len(manual_labels), error4, error4 / len(manual_labels), error5, error5 / len(manual_labels)

def  main():
    MANUAL_PATH = './dataset/annotation_data.csv'
    WORD_FREQ_DICT = './dict/modern_chinese_word_freq.txt'

    BERT_GEN = './data/bert_ss_res.csv'
    BERT_NO_AUTOREGRESSIVE_GEN = './data/bert_no_autoregressive_ss_res.csv'
    BERT_BI_GEN = './data/bert_bi_ss_res.csv'
    BERT_WWM_GEN = './data/bert_wwm_ss_res.csv'
    BERT_WWM_EXT_GEN = './data/bert_wwm_ext_ss_res.csv'
    ERNIE_GEN = './data/ernie_ss_res.csv'
    ELECTRA_GEN = './data/electra_ss_res.csv'
    ROBERTA_WWM_EXT_GEN = './data/roberta_wwm_ext_ss_res.csv'
    MACBERT_GEN = './data/macbert_base_ss_res.csv'
    VECTOR_GEN = './data/vector_ss_res.csv'
    DICT_GEN = './data/dict_ss_res.csv'
    HOWNET_GEN = './data/hownet_ss_res.csv'
    HYBRID_GEN = './data/hybrid_ss_res.csv'

    BERT_OUTPUT = './data/bert_sr_res.csv'
    BERT_NO_AUTOREGRESSIVE_OUTPUT = './data/bert_no_autoregressive_sr_res.csv'
    BERT_WWM_OUTPUT = './data/bert_wwm_sr_res.csv'
    BERT_WWM_EXT_OUTPUT = './data/bert_wwm_ext_sr_res.csv'
    ROEBRTA_WWM_EXT_OUTPUT = './data/roberta_wwm_ext_sr_res.csv'
    MACBERT_OUTPUT = './data/macbert_sr_res.csv'
    ERNIE_OUTPUT = './data/ernie_sr_res.csv'
    ELECTRA_OUTPUT = './data/electra_sr_res.csv'
    VECTOR_OUTPUT = './data/vector_sr_res.csv'
    DICT_OUTPUT = './data/dict_sr_res.csv'
    HOWNET_OUTPUT = './data/hownet_sr_res.csv'
    HYBRID_OUTPUT = './data/hybrid_sr_res.csv'

    BERT_HUMAN = './data/eval/bert_manual_eval.csv'
    DICT_HUMAN = './data/eval/dict_manual_eval.csv'
    VECTOR_HUMAN = './data/eval/vector_manual_eval.csv'
    HOWNET_HUMAN = './data/eval/hownet_manual_eval.csv'
    HYBRID_HUMAN = './data/eval/hybrid_manual_eval.csv'

    word_freq_dict = WORD_FREQ_DICT
    bert_gen = BERT_GEN
    bert_bi_gen = BERT_BI_GEN
    bert_no_autoregressive_gen = BERT_NO_AUTOREGRESSIVE_GEN
    bert_wwm_gen = BERT_WWM_GEN
    bert_wwm_ext_gen = BERT_WWM_EXT_GEN
    ernie_gen = ERNIE_GEN
    electra_gen = ELECTRA_GEN
    roberta_wwm_ext_gen = ROBERTA_WWM_EXT_GEN
    macbert_gen = MACBERT_GEN
    vector_gen = VECTOR_GEN
    dict_gen = DICT_GEN
    hownet_gen = HOWNET_GEN
    hybrid_gen = HYBRID_GEN

    manual_path = MANUAL_PATH
    bert_output = BERT_OUTPUT
    bert_no_autoregressive_output = BERT_NO_AUTOREGRESSIVE_OUTPUT
    bert_wwm_output = BERT_WWM_OUTPUT
    bert_wwm_ext_output = BERT_WWM_EXT_OUTPUT
    roberta_wwm_ext_output = ROEBRTA_WWM_EXT_OUTPUT
    macbert_output = MACBERT_OUTPUT
    ernie_output = ERNIE_OUTPUT
    electra_output = ELECTRA_OUTPUT
    vector_output = VECTOR_OUTPUT
    dict_output = DICT_OUTPUT
    hownet_output = HOWNET_OUTPUT
    hybrid_output = HYBRID_OUTPUT

    origin_words, manual_labels = read_manual_data(manual_path)

    b_g = read_gen_data(bert_gen)
    b_n_a_g = read_gen_data(bert_no_autoregressive_gen)
    b_b_g = read_gen_data(bert_bi_gen)
    b_w_g = read_gen_data(bert_wwm_gen)
    b_w_e_g = read_gen_data(bert_wwm_ext_gen)
    e_g = read_gen_data(ernie_gen)
    el_g = read_gen_data(electra_gen)
    r_w_e_g = read_gen_data(roberta_wwm_ext_gen)
    v_g = read_gen_data(vector_gen)
    d_g = read_gen_data(dict_gen)
    h_g = read_gen_data(hownet_gen)
    m_b_g = read_gen_data(macbert_gen)
    m_g = read_gen_data(hybrid_gen)

    bert_human = BERT_HUMAN
    dict_human = DICT_HUMAN
    vector_human = VECTOR_HUMAN
    hownet_human = HOWNET_HUMAN
    hybrid_human = HYBRID_HUMAN

    bert_pre_words, bert_substitutions = read_sr_data(bert_output)
    bert_wwm_pre_words, bert_substitutions = read_sr_data(bert_wwm_output)
    bert_wwm_ext_pre_words, bert_substitutions = read_sr_data(bert_wwm_ext_output)
    bert_no_autoregressive_pre_words, bert_substitutions = read_sr_data(bert_no_autoregressive_output)
    roberta_pre_words, roberta_substitutions = read_sr_data(roberta_wwm_ext_output)
    macbert_pre_words, macbert_substitutions = read_sr_data(macbert_output)
    ernie_pre_words, ernie_substitutions = read_sr_data(ernie_output)
    electra_pre_words, electra_substitutions = read_sr_data(electra_output)
    vector_pre_words, vector_substitutions = read_sr_data(vector_output)
    dict_pre_words, dict_substitutions = read_sr_data(dict_output)
    hownet_pre_words, hownet_substitutions = read_sr_data(hownet_output)
    hybrid_pre_words, hybrid_substitutions = read_sr_data(hybrid_output)

    bert_human_eval = read_eval_data(bert_human)
    dict_human_eval = read_eval_data(dict_human)
    vector_human_eval = read_eval_data(vector_human)
    hownet_human_eval = read_eval_data(hownet_human)
    hybrid_human_eval = read_eval_data(hybrid_human)

    print('='*30 + 'Exp0' + '='*30 + '\n')
    print('-'*30 + 'BERT' + '-'*30 + '\n')
    changed, human, auto, cons = evaluate_dataset(bert_human_eval, bert_pre_words, origin_words, manual_labels)
    print('changed:' + str(changed) + '\t' + 'human:' + '%.4f'%human + '\t' + 'auto:' + '%.4f'%auto + '\t' + 'cons:' + str(cons))

    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    changed, human, auto, cons = evaluate_dataset(vector_human_eval, vector_pre_words, origin_words, manual_labels)
    print('changed:' + str(changed) + '\t' + 'human:' + '%.4f'%human + '\t' + 'auto:' + '%.4f'%auto + '\t' + 'cons:' + str(cons))

    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    changed, human, auto, cons = evaluate_dataset(dict_human_eval, dict_pre_words, origin_words, manual_labels)
    print('changed:' + str(changed) + '\t' + 'human:' + '%.4f'%human + '\t' + 'auto:' + '%.4f'%auto + '\t' + 'cons:' + str(cons))

    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    changed, human, auto, cons = evaluate_dataset(hownet_human_eval, hownet_pre_words, origin_words, manual_labels)
    print('changed:' + str(changed) + '\t' + 'human:' + '%.4f'%human + '\t' + 'auto:' + '%.4f'%auto + '\t' + 'cons:' + str(cons))

    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    changed, human, auto, cons = evaluate_dataset(hybrid_human_eval, hybrid_pre_words, origin_words, manual_labels)
    print('changed:' + str(changed) + '\t' + 'human:' + '%.4f'%human + '\t' + 'auto:' + '%.4f'%auto + '\t' + 'cons:' + str(cons))

    print('='*30 + 'Exp1' + '='*30 + '\n')
    print('-'*30 + 'BERT-BASE' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(b_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'BERT-BI' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(b_b_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'BERT-NO-AUTOREGRESSIVE' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(b_n_a_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'ERNIE' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(e_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'BERT-WWM' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(b_w_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'BERT-WWM-EXT' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(b_w_e_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'ROBERTA-WWM-EXT' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(r_w_e_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'MACBERT' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(m_b_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'ELECTRA' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(el_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')
    
    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(v_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(d_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(h_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    potential, precision, recall, F_score = evaluate_SS_scores(m_g, manual_labels)
    print('POTN:' + '%.4f'%potential + '\t' + 'PRE:' + '%.4f'%precision + '\t' + 'RE:' + '%.4f'%recall + '\t' + 'F1:' + '%.4f'%F_score + '\n')

    print('='*30 + 'Exp2' + '='*30 + '\n')

    print('-'*30 + 'BERT-BASE' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(bert_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'BERT-NO-AUTOREGRESSIVE' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(bert_no_autoregressive_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'BERT-WWM' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(bert_wwm_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'BERT-WWM-WXT' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(bert_wwm_ext_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'ROBERTA' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(roberta_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'MACBERT' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(macbert_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'ELECTRA' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(electra_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'ERNIE' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(ernie_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(vector_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(dict_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(hownet_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(hybrid_pre_words, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('='*30 + 'Exp3' + '='*30 + '\n')

    word_freq_rank = read_dict(word_freq_dict)

    print('-'*30 + 'BERT' + '-'*30 + '\n')
    no_error, no_error_score, error3a, error3a_score, error3b,  error3b_score, error4, error4_score, error5, error5_score = evaluate_error(word_freq_rank, manual_labels, bert_pre_words, b_g)
    print('no_error:' + '%.4f'%no_error_score + '(%d)'%no_error + '\t' + 'error_1:' + '%.4f'%error3a_score +'(%d)'%error3a + '\t' + 'error_2:' + '%.4f'%error3b_score + '(%d)'%error3b + '\t' + 'error_3:' + '%.4f'%error4_score + '(%d)'%error4 + '\t' + 'error5_score:' + '%.4f'%error5_score + '(%d)'%error5 + '\n')
    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    no_error, no_error_score, error3a, error3a_score, error3b,  error3b_score, error4, error4_score, error5, error5_score = evaluate_error(word_freq_rank, manual_labels, vector_pre_words, v_g)
    print('no_error:' + '%.4f'%no_error_score + '(%d)'%no_error + '\t' + 'error_1:' + '%.4f'%error3a_score +'(%d)'%error3a + '\t' + 'error_2:' + '%.4f'%error3b_score + '(%d)'%error3b + '\t' + 'error_3:' + '%.4f'%error4_score + '(%d)'%error4 + '\t' + 'error5_score:' + '%.4f'%error5_score + '(%d)'%error5 + '\n')
    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    no_error, no_error_score, error3a, error3a_score, error3b,  error3b_score, error4, error4_score, error5, error5_score = evaluate_error(word_freq_rank, manual_labels, dict_pre_words, d_g)
    print('no_error:' + '%.4f'%no_error_score + '(%d)'%no_error + '\t' + 'error_1:' + '%.4f'%error3a_score +'(%d)'%error3a + '\t' + 'error_2:' + '%.4f'%error3b_score + '(%d)'%error3b + '\t' + 'error_3:' + '%.4f'%error4_score + '(%d)'%error4 + '\t' + 'error5_score:' + '%.4f'%error5_score + '(%d)'%error5 + '\n')
    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    no_error, no_error_score, error3a, error3a_score, error3b,  error3b_score, error4, error4_score, error5, error5_score = evaluate_error(word_freq_rank, manual_labels, hownet_pre_words, h_g)
    print('no_error:' + '%.4f'%no_error_score + '(%d)'%no_error + '\t' + 'error_1:' + '%.4f'%error3a_score +'(%d)'%error3a + '\t' + 'error_2:' + '%.4f'%error3b_score + '(%d)'%error3b + '\t' + 'error_3:' + '%.4f'%error4_score + '(%d)'%error4 + '\t' + 'error5_score:' + '%.4f'%error5_score + '(%d)'%error5 + '\n')
    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    no_error, no_error_score, error3a, error3a_score, error3b,  error3b_score, error4, error4_score, error5, error5_score = evaluate_error(word_freq_rank, manual_labels, hybrid_pre_words, m_g)
    print('no_error:' + '%.4f'%no_error_score + '(%d)'%no_error + '\t' + 'error_1:' + '%.4f'%error3a_score +'(%d)'%error3a + '\t' + 'error_2:' + '%.4f'%error3b_score + '(%d)'%error3b + '\t' + 'error_3:' + '%.4f'%error4_score + '(%d)'%error4 + '\t' + 'error5_score:' + '%.4f'%error5_score + '(%d)'%error5 + '\n')
    no_hownet_bert, _ = read_sr_data('./test/data/nohownet/bert_sr_res_no_hownet.csv')
    no_hownet_embedding, _ = read_sr_data('./test/data/nohownet/vector_sr_res_no_hownet.csv')
    no_hownet_dict, _ = read_sr_data('./test/data/nohownet/dict_sr_res_no_hownet.csv')
    no_hownet_hownet, _ = read_sr_data('./test/data/nohownet/hownet_sr_res_no_hownet.csv')
    no_hownet_hybrid, _ = read_sr_data('./test/data/nohownet/hybrid_sr_res_no_hownet.csv')

    no_bert_bert, _ = read_sr_data('./test/data/nobert/bert_sr_res_no_bert.csv')
    no_bert_embedding, _ = read_sr_data('./test/data/nobert/vector_sr_res_no_bert.csv')
    no_bert_dict, _ = read_sr_data('./test/data/nobert/dict_sr_res_no_bert.csv')
    no_bert_hownet, _ = read_sr_data('./test/data/nobert/hownet_sr_res_no_bert.csv')
    no_bert_hybrid, _ = read_sr_data('./test/data/nobert/hybrid_sr_res_no_bert.csv')

    no_embedding_bert, _ = read_sr_data('./test/data/noembedding/bert_sr_res_no_embedding.csv')
    no_embedding_embedding, _ = read_sr_data('./test/data/noembedding/vector_sr_res_no_embedding.csv')
    no_embedding_dict, _ = read_sr_data('./test/data/noembedding/dict_sr_res_no_embedding.csv')
    no_embedding_hownet, _ = read_sr_data('./test/data/noembedding/hownet_sr_res_no_embedding.csv')
    no_embedding_hybrid, _ = read_sr_data('./test/data/noembedding/hybrid_sr_res_no_embedding.csv')

    no_freq_bert, _ = read_sr_data('./test/data/nofreq/bert_sr_res_no_freq.csv')
    no_freq_embedding, _ = read_sr_data('./test/data/nofreq/vector_sr_res_no_freq.csv')
    no_freq_dict, _ = read_sr_data('./test/data/nofreq/dict_sr_res_no_freq.csv')
    no_freq_hownet, _ = read_sr_data('./test/data/nofreq/hownet_sr_res_no_freq.csv')
    no_freq_hybrid, _ = read_sr_data('./test/data/nofreq/hybrid_sr_res_no_freq.csv')

    no_chnum_bert, _ = read_sr_data('./test/data/nochnum/bert_sr_res_no_chnum.csv')
    no_chnum_embedding, _ = read_sr_data('./test/data/nochnum/vector_sr_res_no_chnum.csv')
    no_chnum_dict, _ = read_sr_data('./test/data/nochnum/dict_sr_res_no_chnum.csv')
    no_chnum_hownet, _ = read_sr_data('./test/data/nochnum/hownet_sr_res_no_chnum.csv')
    no_chnum_hybrid, _ = read_sr_data('./test/data/nochnum/hybrid_sr_res_no_chnum.csv')

    print('='*30 + 'Exp4' + '='*30 + '\n')
    print('='*30 + 'NOHOWNET' + '='*30 + '\n')
    print('-'*30 + 'BERT' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_hownet_bert, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_hownet_embedding, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_hownet_dict, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_hownet_hownet, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_hownet_hybrid, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('='*30 + 'NOBERT' + '='*30 + '\n')
    print('-'*30 + 'BERT' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_bert_bert, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_bert_embedding, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_bert_dict, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_bert_hownet, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_bert_hybrid, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('='*30 + 'NOEMBEDDING' + '='*30 + '\n')
    print('-'*30 + 'BERT' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_embedding_bert, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_embedding_embedding, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_embedding_dict, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_embedding_hownet, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_embedding_hybrid, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

    print('='*30 + 'NOFREQ' + '='*30 + '\n')
    print('-'*30 + 'BERT' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_freq_bert, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Embedding' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_freq_embedding, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Linguistic' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_freq_dict, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Sememe' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_freq_hownet, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')
    print('-'*30 + 'Hybrid' + '-'*30 + '\n')
    precision, accuracy, changed_proportion = evaluate_pipeline_scores(no_freq_hybrid, origin_words, manual_labels)
    print('PRE:' + '%.4f'%precision + '\t' + 'ACC:' + '%.4f'%accuracy + '\n')

if __name__=='__main__':
    main()