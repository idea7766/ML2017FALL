wget 'https://www.dropbox.com/s/leomtjns7ikzvwu/model_re_bst.h5?dl=1' -O 'model_re_bst.h5'
wget 'https://www.dropbox.com/s/o6bpuk0v5fnj1vf/model_ta.h5?dl=1' -O 'model_ta.h5'
wget 'https://www.dropbox.com/s/08mvtifqe53bdsi/model_x.h5?dl=1' -O 'model_x.h5'
python vote.py $@