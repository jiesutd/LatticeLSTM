Chinese NER Using Lattice LSTM
====

Lattice LSTM for Chinese NER. Character based LSTM with Lattice embeddings as input.

Models and results can be found at our ACL 2018 paper [Chinese NER Using Lattice LSTM](https://arxiv.org/pdf/1805.02023.pdf). It achieves 93.18% F1-value on MSRA dataset, which is the state-of-the-art result on Chinese NER task.

Details will be updated soon.

Requirement:
======
	Python: 2.7   
	PyTorch: 0.3.0 
(for PyTorch 0.3.1, please refer [issue#8](https://github.com/jiesutd/LatticeLSTM/issues/8) for a slight modification.)

Input format:
======
CoNLL format (prefer BIOES tag scheme), with each character its label for one line. Sentences are splited with a null line.

	美	B-LOC
	国	E-LOC
	的	O
	华	B-PER
	莱	I-PER
	士	E-PER

	我	O
	跟	O
	他	O
	谈	O
	笑	O
	风	O
	生	O 

Pretrained Embeddings:
====
The pretrained character and word embeddings are the same with the embeddings in the baseline of [RichWordSegmentor](https://github.com/jiesutd/RichWordSegmentor)

Character embeddings: [gigaword_chn.all.a2b.uni.ite50.vec](https://pan.baidu.com/s/1pLO6T9D)

Word(Lattice) embeddings: [ctb.50d.vec](https://pan.baidu.com/s/1pLO6T9D)

How to run the code?
====
1. Download the character embeddings and word embeddings and put them in the `data` folder.
2. Modify the `run_main.py` or `run_demo.py` by adding your train/dev/test file directory.
3. `sh run_main.py` or `sh run_demo.py`


Resume NER data 
====
Crawled from the Sina Finance, it includes the resumes of senior executives from listed companies in the Chinese stock market. Details can be found in our paper.


Cite: 
========
Please cite our ACL 2018 paper:

    @article{zhang2018chinese,  
     title={Chinese NER Using Lattice LSTM},  
     author={Yue Zhang and Jie Yang},  
     booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
     year={2018}  
    }