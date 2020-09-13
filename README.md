<p align="center"><img src="./docs/img/logo.png" width = "250"  alt="Chinese-LS Logo"/></p>

English|[简体中文](README.zh.md)

## What is Chinese-LS?

Lexical simplification (LS) aims to replace complex words in a given sentence with their simpler alternatives of equivalent meaning. Chinese-LS is the first attempt in the field of Chinese Lexical Simplification. It includes a high-quality benchmark dataset and five baseline approaches: 

1. Synonym dictionary-based approach
2. Word embedding-based approach
3. Pretrained language model-based approach
4. Sememe-based approach
5. Hybrid approach

The entire framework of Chinese-LS is shown below:

<p align="center"><img src="docs/img/Chinese_lexical_simplification_system.png" width = "500"  alt="Chinese-LS Framework"/></p>

## Contact

Email: luxinyu12345@foxmail.com

## Quick start

### Requirements

- Python==3.7.6
- transformers==2.9.0
- numpy==1.18.1
- jieba==0.42.1
- torch==1.4.0
- OpenHowNet==0.0.1a11
- gensim==3.8.2

The complete ```requirements.txt``` can been seen [here](requirements.txt).

### Preparations

#### Download Pretrained Models

Chinese-LS uses the following pretrained models:

- Word2Vec model: [Chinese-Word-Vector](https://github.com/Embedding/Chinese-Word-Vectors) (Mixed-large)
- BERT-base, Chinese ([transformers](https://huggingface.co/bert-base-chinese)) 

Please place the models under the ```/model``` directory after downloading.

### Run

We have already executed the codes for you and intermediate results can be found in ```/data```.

If you want to run the codes for reproduction, please execute them in the following order: 

#### Generate

1. Synonym dictionary based-approach

	Run ```dict_generate.py```
	
2. Word embedding based-approach

	Run ```vector_generate.py```

3. Pretrained language model based-approach

	Run ```bert_generate.py```

4. Sememe based-approach

	Run ```hownet_generate.py```

5. Hybrid approach

	Run ```hybrid_approach.py```

#### Select

Run ```substitute_selection.py```

#### Rank

Run ```substitute_ranking.py```

### Experiment

Run ```experiment.py```

## Citation

```

```

## License

Chinese-LS is under the [Apache License, Version 2.0](https://github.com/luxinyu1/Chinese-LS/blob/master/LICENSE).