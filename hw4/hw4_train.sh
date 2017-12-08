wget 'https://www.dropbox.com/s/ck198ssfjljkocl/dictionary?dl=1' -O 'dictionary'
wget 'https://www.dropbox.com/s/7drtjeuj4qs4qqc/tokenizer?dl=1' -O 'tokenizer'
wget 'https://www.dropbox.com/s/2x7fv9n1rv27k2d/word_vec?dl=1' -O 'word_vec'
python train.py $@