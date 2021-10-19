<p align="center"><img src="./docs/img/logo.png" width = "250"  alt="Chinese-LS Logo"/></p>

English|[简体中文](README.zh.md)

## What is Chinese-LS?

Lexical simplification (LS) aims to replace complex words in a given sentence with their simpler alternatives of equivalent meaning. Chinese-LS is the first attempt in the field of Chinese Lexical Simplification. It includes a high-quality benchmark [dataset](./dataset/annotation_data.csv) and five baseline approaches: 

- Synonym dictionary-based approach

- Word embedding-based approach

- Pretrained language model-based approach

- Sememe-based approach

- Hybrid approach

The entire framework of Chinese-LS is shown below:

<p align="center"><img src="docs/img/Chinese_lexical_simplification_system.png" width = "700"  alt="Chinese-LS Framework"/></p>

## Quick start

### Requirements

- Python==3.7.6
- transformers==3.5.0
- numpy==1.18.1
- jieba==0.42.1
- torch==1.4.0
- OpenHowNet==0.0.1a11
- gensim==3.8.2

You can find the complete requirements [here](requirements.txt).

### Preparations

#### Download Pretrained Models

Chinese-LS uses the following pretrained models:

- Word2Vec model: [Chinese-Word-Vector](https://github.com/Embedding/Chinese-Word-Vectors) (Mixed-large)
- BERT-base, Chinese ([huggingface](https://huggingface.co/bert-base-chinese))
- macbert-base, Chinese ([huggingface](https://huggingface.co/hfl/chinese-macbert-base))
- bert-wwm-ext, Chinese ([huggingface](https://huggingface.co/hfl/chinese-bert-wwm-ext)) 
- roberta-wwm-ext, Chinese ([huggingface](https://huggingface.co/hfl/chinese-roberta-wwm-ext)) 
- ERNIE ([PaddlePaddle](https://baidu-nlp.bj.bcebos.com/ERNIE_stable-1.0.1.tar.gz))

Please place the models under the ```./model``` directory after downloading.

### Run

We have already executed the codes for you and intermediate results can be found in ```./data```.

You could check out the details of codes and algorithms from our paper: [Chinese Lexical Simplification](https://ieeexplore.ieee.org/abstract/document/9439908)

If you want to run the codes for reproduction, please execute them in the following order: 

#### Generate

1. Synonym dictionary based-approach

	Run ```dict_generate.py```
	
2. Word embedding based-approach

	Run ```vector_generate.py```

3. Pretrained language model based-approach

	Run ```bert_generate.sh```

4. Sememe based-approach

	Run ```hownet_generate.py```

5. Hybrid approach

	Run ```hybrid_approach.py```

#### Select

Run ```substitute_selection.py```

#### Rank

Run ```substitute_ranking.py```

### Experiments

Chinese-LS designs 5 experiments to evaluate the quality of our dataset and the performance of five approaches. You could get the experiment results through running ```experiment.py```.

## Citation

```
@article{qiang2021chinese,
    title={Chinese Lexical Simplification},
    author={Qiang, Jipeng and Lu, Xinyu and Li, Yun and Yuan, Yun-Hao and Wu, Xindong},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
    year={2021},
    volume={29},
    pages={1819-1828},
    doi={10.1109/TASLP.2021.3078361},
    publisher={IEEE}
}
```

## Contact

This repo may still contain bugs and we are working on improving the reproductivity. Welcome to open an issue or submit a Pull Request to report/fix the bugs.

Email: luxinyu12345@foxmail.com

## License

Chinese-LS is under the [Apache License, Version 2.0](https://github.com/luxinyu1/Chinese-LS/blob/master/LICENSE).