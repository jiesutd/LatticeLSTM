python main.py --status train \
		--train ../data/onto4ner.cn/train.char.bmes \
		--dev ../data/onto4ner.cn/dev.char.bmes \
		--test ../data/onto4ner.cn/test.char.bmes \
		--savemodel ../data/onto4ner.cn/saved_model \


# python main.py --status decode \
# 		--raw ../data/onto4ner.cn/test.char.bmes \
# 		--savedset ../data/onto4ner.cn/saved_model \
# 		--loadmodel ../data/onto4ner.cn/saved_model.13.model \
# 		--output ../data/onto4ner.cn/raw.out \
