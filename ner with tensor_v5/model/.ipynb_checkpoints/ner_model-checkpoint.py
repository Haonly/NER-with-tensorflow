import numpy as np
import os
import tensorflow as tf

from .data_utils import minibatches,pad_sequences,get_chunks
from .general_utils import Progbar
from .base_model import BaseModel

class NERModel(BaseModel):
    
    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
    
    def add_placeholders(self):

        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],name="word_ids")

        # shape = (batch size) 20
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],name="labels")


        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],name="lr")
    
    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        # perform padding of the given data
        
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths = pad_sequences(word_ids, 0)
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)
      
        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths
        
    def add_word_embeddings_op(self):
        """ embedding_lookup """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:  # embedding이 없을 경우.
                self.logger.info("WARNING: randomly initializing word vectors")
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else: ## 여기가 작동.
                _word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,self.word_ids, name="word_embeddings")

            
        """ char embedding """ 
        with tf.variable_scope("chars"):
                
            if self.config.use_chars:
                if self.config.char_method=='BiLSTM':
                    print('Character - BiLSTM')
                    # get char embeddings matrix
                    _char_embeddings = tf.get_variable(name="_char_embeddings",dtype=tf.float32,shape=[self.config.nchars, self.config.dim_char])
                    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,self.char_ids, name="char_embeddings")

                    s = tf.shape(char_embeddings) # 텐서의 구조를 반환
                    char_embeddings = tf.reshape(char_embeddings,shape=[s[0]*s[1], s[-2], self.config.dim_char]) ## 3차원 텐서로 변경
                    word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])
                    
                    cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,state_is_tuple=True) # 100
                    cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,state_is_tuple=True) #100
                    _output = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, char_embeddings,sequence_length=word_lengths, dtype=tf.float32)

                    # read and concat output
                    _, ((_, output_fw), (_, output_bw)) = _output
                    output = tf.concat([output_fw, output_bw], axis=-1)

                    output = tf.reshape(output,shape=[s[0], s[1], 2*self.config.hidden_size_char]) 
                    word_embeddings = tf.concat([word_embeddings, output], axis=-1) # (?,?,500)
                    
                elif self.config.char_method=='CNN':
                    print('Character - CNN')
                    _char_embeddings = tf.get_variable(name="_char_embeddings",dtype=tf.float32,shape=[self.config.nchars, self.config.dim_char])
                    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,self.char_ids, name="char_embeddings")
                
                    s = tf.shape(char_embeddings) # 텐서의 구조를 반환
                    char_embeddings = tf.reshape(char_embeddings,shape=[s[0]*s[1], s[-2], self.config.dim_char]) ## 3차원 텐서로 변경
                    
#                     word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                    char_feature=tf.expand_dims(char_embeddings,-1) # 텐서 차원 확장
               
                    features=[]

                    for filterSize in [2,3,4,5]:
                        char_features=tf.pad(char_feature,[[0,0],[filterSize-1,filterSize-1],[0,0],[0,0]])
                        csr=tf.layers.conv2d(inputs=char_features,filters=30,kernel_size=[filterSize,100])
             
                        feature1=tf.reduce_max(csr,axis=1)
                        feature2=tf.squeeze(feature1)
                        feature3=tf.reshape(feature2,[-1,s[1],30]) 

                        features.append(feature3)

                    layer_out=tf.concat(features,axis=2)

                    logits=tf.layers.dense(inputs=layer_out,units=30,activation=None,
                                          kernel_initializer=tf.initializers.variance_scaling(scale=2.0,mode='fan_in',
                                                                                             distribution='normal'))
              
                    word_embeddings=tf.concat([word_embeddings,logits],axis=-1) ## (?, ?, 330)
                 
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
        
        
    def add_logits_op(self):
        with tf.variable_scope("bi-lstm"):
            """RNN"""
#            cell_fw=tf.contrib.rnn.BasicRNNCell(self.config.hidden_size_lstm)
#            cell_bw=tf.contrib.rnn.BasicRNNCell(self.config.hidden_size_lstm)
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
#            cell_fw = tf.contrib.rnn.GRUCell(self.config.hidden_size_lstm)
#            cell_bw = tf.contrib.rnn.GRUCell(self.config.hidden_size_lstm)
            
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1) # (?,?,600)
            output = tf.nn.dropout(output, self.dropout)
            
        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
    
    """ crf를 사용하지 않으면 """
    def add_pred_op(self):
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),tf.int32) # 최대값을 label로 추출
    
    def add_loss_op(self):
        if self.config.use_crf: ## crf 사용
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

    def build(self):
        
        self.add_placeholders() #placeholder 선언하고
        self.add_word_embeddings_op() # embedding 
        self.add_logits_op() # 모델 구성
        self.add_pred_op() # softmax를 사용했을 경우 argmax 해주는 함수
        self.add_loss_op() # cost function 사용
        
        self.add_train_op(self.config.lr_method, self.lr, self.loss) # BaseModel 함수 상속. optimizer 방법과 learning rate, cost function을 전달.
        self.initialize_session() # session open
    
    def predict_batch(self, words):
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)
            
        if self.config.use_crf: # CRF 사용
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=fd) # 학습.

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else: # SoftMax 사용
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths
    
    def run_epoch(self,train,dev,epoch):
        
        batch_size=self.config.batch_size # 20
        nbatches=(len(train)+batch_size-1)//batch_size # //는 나눗셈을 의미하며 결과가 int로 나옴.
        prog=Progbar(target=nbatches)
        
        for i,(words,labels) in enumerate(minibatches(train,batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,self.config.dropout)
        
            _, train_loss= self.sess.run([self.train_op, self.loss], feed_dict=fd)

            prog.update(i + 1, [("train loss", train_loss)])

        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]
    
    def run_evaluate(self, test):

        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds
