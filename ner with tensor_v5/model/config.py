
import os
from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors,load_vocab,get_processing_word

class Config():
    def __init__(self,load=True):
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        
        self.logger=get_logger(self.path_log)
        
        if load:
            self.load()
    
    def load(self):
        self.vocab_words=load_vocab(self.filename_words) # 단어 load
        self.vocab_tags=load_vocab(self.filename_tags) # 태그 load
        self.vocab_chars=load_vocab(self.filename_chars) # char load
        
        self.processing_word=get_processing_word(self.vocab_words,self.vocab_chars,lowercase=True,chars=self.use_chars)
        self.processing_tag=get_processing_word(self.vocab_tags,lowercase=False,allow_unk=False)
        self.embeddings=(get_trimmed_glove_vectors(self.filename_trimmed) if self.use_pretrained else None)
        
        self.nwords=len(self.vocab_words)
        self.nchars=len(self.vocab_chars)
        self.ntags=len(self.vocab_tags)
        
    
    dir_output = "results/test/"
    dir_model  = dir_output + "model.weights/"
    path_log   = dir_output + "log.txt"
    
    dim_word=300 # 단어는 300 차원
    dim_char=100 # character는 100 차원
    
    filename_glove='./glove.6B.300d.txt'.format(dim_word) ## glove vector를 가지고 와서
    filename_trimmed='./glove.6B.300d.trimmed.npz'.format(dim_word) # npz 파일로 저장한다.
    use_pretrained=True
    
    filename_train='./train.txt'
    filename_dev='./valid.txt'
    filename_test='./test.txt'
    
#     filename_dev=filename_test=filename_train='./test.txt' # 시간이 너무 오래 걸려 이걸로 test 했음.
    
    max_iter=None
    
    filename_words='./words.txt'
    filename_tags='./tags.txt'
    filename_chars='./chars.txt'
    
    train_embeddings=False
    nepochs=15
    dropout=0.5
    batch_size=20
    lr_method='adam'
    lr=0.001
    lr_decay=0.9
    nepoch_no_imprv=3
    
    hidden_size_char=100
    hidden_size_lstm=300
    
    use_crf=False
    use_chars=True
    char_method='CNN' # CNN or BiLSTM
