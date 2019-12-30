PYBIN := python.exe

.PHONY : corpus clean clean_models

lemmatizer:
	cd app; $(PYBIN) lemmatizer.py -v
test_lemmatizer:
	cd app; $(PYBIN) lemmatizer.py test -v

tokenizer:
	cd app; $(PYBIN) tokenizer.py -v
test_tokenizer:
	cd app; $(PYBIN) tokenizer.py test -v

parser:
	cd app; $(PYBIN) parser.py -v
test_parser:
	cd app; $(PYBIN) parser.py test -v

test_aes_ML:
	cd app/auto_essay_score; $(PYBIN) aes_ML.py

test_aes_NN:
	cd app/auto_essay_score; $(PYBIN) aes_NN.py

corpus:
	mkdir -p corpus model
	cd corpus && [ -f dic_ec.rar ] || (wget http://nlp.nju.edu.cn/MT_Lecture/dic_ec.rar && 7z x dic_ec.rar)
	cd corpus && [ -f dic_ec.rar ] || (wget http://nlp.nju.edu.cn/MT_Lecture/dic_ce.rar && 7z x dic_ce.rar)
	cp -Ruv app/auto_essay_score/ProblemDescription/essay_data corpus

clean:
	rm -rf app/__pycache__

clean_models:
	rm -rf model/*
