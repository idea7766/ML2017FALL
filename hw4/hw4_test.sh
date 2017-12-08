wget 'https://www.dropbox.com/s/s45wav934ydqqsz/model_bi_lstm.h5?dl=0' -O 'model_bi_lstm.h5'
wget 'https://www.dropbox.com/s/7drtjeuj4qs4qqc/tokenizer?dl=1' -O 'tokenizer'
wget 'https://www.dropbox.com/s/2x7fv9n1rv27k2d/word_vec?dl=1' -O 'word_vec'
python predict.py $@