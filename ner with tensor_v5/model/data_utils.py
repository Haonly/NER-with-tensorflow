
import numpy as np
import os

UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = "aa"

class CoNLLDataset(object):
    def __init__(self,filename,processing_word=None,processing_tag=None,
                max_iter=None):
        
        self.filename=filename
        self.processing_word=processing_word
        self.processing_tag=processing_tag
        self.max_iter=max_iter
        self.length=None
        
    def __iter__(self):
        niter=0
        
        with open(self.filename) as f:
            words,tags=[],[]
            
            for line in f:
                line=line.strip()
                if(len(line)==0 or line.startswith('-DOCSTART-')):
                    if len(words)!=0:
                        niter+=1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words,tags
                        words,tags=[],[]
                else:
                    ls=line.split(' ')
                    word,tag=ls[0],ls[-1]
                    if self.processing_word is not None:
                        word=self.processing_word(word)
                    if self.processing_tag is not None:
                        tag=self.processing_tag(tag)
                    words+=[word]               
                    tags+=[tag]
    
    def __len__(self):
        if self.length is None:
            self.length=0
            for _ in self:
                self.length+=1
        return self.length
    
##  word와 tag만 load

def get_vocabs(dataset):
    
    print('Building vocab...')
    vocab_words=set()
    vocab_tags=set()
    
    for dataset in dataset:
        for words,tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print('-done. {} tokens'.format(len(vocab_words)))
    return vocab_words,vocab_tags

def get_char_vocab(dataset):
    vocab_char=set()
    
    for words,_ in dataset:
        for word in words:
            vocab_char.update(word)
    
    return vocab_char

def get_glove_vocab(filename):
    print('Building vocab...')
    vocab=set()
    
    with open(filename,encoding='utf-8') as f:
        for line in f:
            word=line.strip().split(' ')[0]
            vocab.add(word)
    print('-done. {} tokens'.format(len(vocab)))
    return vocab

def write_vocab(vocab,filename):
    print('Writing vocab...')
    with open(filename,'w') as f:
        for i,word in enumerate(vocab):
            if i!=len(vocab)-1:
                f.write('{}\n'.format(word))
            else:
                f.write(word)
    print('- done. {} tokens'.format(len(vocab)))

def load_vocab(filename):
    try:
        d=dict()
        with open(filename) as f:
            for idx,word in enumerate(f):
                word=word.strip()
                d[word]=idx
    except IOError:
        raise MyIOError(filename)
    return d

def export_trimmed_glove_vectors(vocab,glove_filename,trimmed_filename,dim):
    
    embeddings=np.zeros([len(vocab),dim]) # 300차원, zero로 벡터 초기화
    with open(glove_filename,encoding='utf-8') as f:
        for line in f:
            line=line.strip().split(' ')
            word=line[0] # 단어
            embedding=[float(x) for x in line[1:]] # embeddings
            if word in vocab: # glove 단어가 vocab 단어 안에 있으면.
                word_idx=vocab[word] 
                embeddings[word_idx]=np.asarray(embedding) # len 7398
 
    np.savez_compressed(trimmed_filename,embeddings=embeddings) # embedding 파일만 저장

def get_trimmed_glove_vectors(filename):
    
    try:
        with np.load(filename) as data:
            return data['embeddings']
    except IOError:
        raise MyIOError(filename)

def get_processing_word(vocab_words=None, vocab_chars=None,lowercase=False,chars=False,allow_unk=True):
    
    def f(word):
        if vocab_chars is not None and chars==True:
            char_ids=[]
            for char in word:
                if char in vocab_chars:
                    char_ids+=[vocab_chars[char]]
        
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM
        
        if vocab_words is not None:
            if word in vocab_words:
                word=vocab_words[word]
            else:
                if allow_unk:
                    word=vocab_words[UNK]
                else:
                    raise Exception('Unknow key is not allowed. Check that your vocab (tags?) is correct')
        
        if vocab_chars is not None and chars==True:
            return char_ids,word
        else:
            return word
    
    return f

def _pad_sequences(sequences,pad_tok,max_length):
    
    sequence_padded,sequence_length=[],[]
    
    for seq in sequences:
        seq=list(seq)
        seq_=seq[:max_length]+[pad_tok]*max(max_length-len(seq),0)
        sequence_padded+=[seq_]
        sequence_length+=[min(len(seq),max_length)]
    return sequence_padded,sequence_length

def pad_sequences(sequences,pad_tok,nlevels=1):
    if nlevels==1:
        max_length=max(map(lambda x:len(x),sequences))
        sequence_padded,sequence_length=_pad_sequences(sequences,pad_tok,max_length)
    elif nlevels==2:
        max_length_word=max([max(map(lambda x:len(x),seq)) for seq in sequences])
        sequence_padded,sequence_length=[],[]
        
        for seq in sequences:
            sp,sl=_pad_sequences(seq,pad_tok,max_length_word)
            sequence_padded+=[sp]
            sequence_length+=[sl]
        
        max_length_sentence=max(map(lambda x: len(x), sequences))
        sequence_padded,_=_pad_sequences(sequence_padded,[pad_tok]*max_length_word,max_length_sentence)
        sequence_length,_=_pad_sequences(sequence_length,0,max_length_sentence)
    
    return sequence_padded,sequence_length

def minibatches(data,minibatch_size):
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch

def get_chunk_type(tok,idx_to_tag):
    
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type

def get_chunks(seq,tags):
    
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
