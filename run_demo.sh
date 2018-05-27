python main.py --status train \
		--train ../data/onto4ner.cn/demo.train.char \
		--dev ../data/onto4ner.cn/demo.dev.char \
		--test ../data/onto4ner.cn/demo.test.char \
		--savemodel ../data/onto4ner.cn/demo \


# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/demo.test.char \
# 		--savedset ../data/onto4ner.cn/demo.dset \
# 		--loadmodel ../data/onto4ner.cn/demo.0.model \
# 		--output ../data/onto4ner.cn/demo.raw.out \
