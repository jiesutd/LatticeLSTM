Chinese NER Using Lattice LSTM
====

Lattice LSTM for Chinese NER. Character based LSTM with Lattice embeddings as input.

Details will be updated soon.

Input format:
======
CoNLL format, with each character its label for one line. Sentences are splited with a null line.

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


Requirement:
======
	Python: 2.7   
	PyTorch: 0.3


Cite: 
========
Please cite our ACL 2018 paper:

    @article{zhang2018chinese,  
     title={Chinese NER Using Lattice LSTM},  
     author={Yue Zhang and Jie Yang},  
     booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL)},
     year={2018}  
    }