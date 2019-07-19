#encoding=utf-8
import os
root_dir = '/home1/liushaoweihua/jupyterlab/datagrand/'
os.chdir(root_dir)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gensim
import numpy as np
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')

from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.engine.base_layer import Layer
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional, Dropout, Flatten
from keras.models import Model
from keras_layer_normalization import LayerNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

class Metrics(object):
    @staticmethod
    def f1(y_true, y_pred):
        def recall(y_true, y_pred):
            """Recall metric.
            Only computes a batch-wise average of recall.
            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.
            Only computes a batch-wise average of precision.
            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

class ClassificationReport(Callback):
    def __init__(self,target_names,**kwargs):
        self.target_names = {str(i):j for i,j in target_names.items()}
        super(ClassificationReport,self).__init__(**kwargs)
        
    def on_train_begin(self,logs={}):
        pass
    
    def on_epoch_end(self,epoch,logs={}):
        global val_pred,val_true
        val_pred = np.argmax(self.model.predict(self.validation_data[0]),axis=-1)
        val_pred = [str(item) for item in np.hstack(val_pred)]
        val_true = np.argmax(self.validation_data[1],axis=-1)
        val_true = [str(item) for item in np.hstack(val_true)]
        print(classification_report(y_true=val_true, y_pred=val_pred, target_names=self.target_names))
        return
    
class Attention(Layer):

    def __init__(self, multiheads, head_dim, mask_right=False, **kwargs):
        """
        # 参数：
        #    - multiheads: Attention的数目
        #    - head_dim: Attention Score的维度
        #    - mask_right: Position-wise Mask，在Encoder时不使用，在Decoder时使用
        """
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        super(Attention, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1],
                self.output_dim)  # shape=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),  # input_shape[0] -> Q_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),  # input_shape[1] -> K_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),  # input_shape[2] -> V_seq
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='add'):
        """
        # 需要对sequence进行Mask以忽略填充部分的影响，一般将填充部分设置为0。
        # 由于Attention中的Mask要放在softmax之前，则需要给softmax层输入一个非常大的负整数，以接近0。
        # 参数：
        #    - inputs: 输入待mask的sequence
        #    - seq_len: shape=[batch_size,1]或[batch_size,]
        #    - mode: mask的方式，'mul'时返回的mask位置为0，'add'时返回的mask位置为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
        """
        if seq_len == None:
            return inputs
        else:
            # seq_len[:,0].shape=[batch_size,1]
            # short_sequence_length=K.shape(inputs)[1]：较短的sequence_length，如K_sequence_length，V_sequence_length
            mask = K.one_hot(indices=seq_len[:, 0], num_classes=K.shape(inputs)[
                1])  # mask.shape=[batch_size,short_sequence_length],mask=[[0,0,0,0,1,0,0,..],[0,1,0,0,0,0,0...]...]
            mask = 1 - K.cumsum(mask,
                                axis=1)  # mask.shape=[batch_size,short_sequence_length],mask=[[1,1,1,1,0,0,0,...],[1,0,0,0,0,0,0,...]...]
            # 将mask增加到和inputs一样的维度，目前仅有两维[0],[1]，需要在[2]上增加维度
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            # mask.shape=[batch_size,short_sequence_length,1,1]
            if mode == 'mul':
                # Element-wise multiply：直接做按位与操作
                # return_shape = inputs.shape
                # 返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                # masked_i的值为0
                return inputs * mask
            elif mode == 'add':
                # Element-wise add：直接做按位加操作
                # return_shape = inputs.shape
                # 返回值：[[seq_element_1,seq_element_2,...,masked_1,masked_2,...],...]，其中seq_element_i,masked_i的维度均为2维
                # masked_i的值为一个非常大的负数，在softmax下为0。由于attention的mask是在softmax之前，所以要用这种方式执行
                return inputs - (1 - mask) * 1e12

    def call(self, QKVs):
        """
        # 参照keras.engine.base_layer的call方法。
        # 1. Q',K',V' = Q .* WQ_i,K .* WK_i,V .* WV_i
        # 2. head_i = Attention(Q',K',V') = softmax((Q' .* K'.T)/sqrt(d_k)) .* V
        # 3. MultiHead(Q,K,V) = Concat(head_1,...,head_n)
        # 参数
            - QKVs：[Q_seq,K_seq,V_seq]或[Q_seq,K_seq,V_seq,Q_len,V_len]
                -- Q_seq.shape = [batch_size,Q_sequence_length,Q_embedding_dim]
                -- K_seq.shape = [batch_size,K_sequence_length,K_embedding_dim]
                -- V_seq.shape = [batch_size,V_sequence_length,V_embedding_dim]
                -- Q_len.shape = [batch_size,1],如：[[7],[5],[3],...]
                -- V_len.shape = [batch_size,1],如：[[7],[5],[3],...]
        # 返回
            -
        """
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(QKVs) == 3:
            Q_seq, K_seq, V_seq = QKVs
            Q_len, V_len = None, None
        elif len(QKVs) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = QKVs
        # 对Q、K、V做线性变换，以Q为例进行说明
        # Q_seq.shape=[batch_size,Q_sequence_length,Q_embedding_dim]
        # self.WQ.shape=[Q_embedding_dim,self.output_dim]=[Q_embedding_dim,self.multiheads*self.head_dim]
        Q_seq = K.dot(Q_seq,
                      self.WQ)  # Q_seq.shape=[batch_size,Q_sequence_length,self.output_dim]=[batch_size,Q_sequence_length,self.multiheads*self.head_dim]
        Q_seq = K.reshape(Q_seq, shape=(-1, K.shape(Q_seq)[1], self.multiheads,
                                        self.head_dim))  # Q_seq.shape=[batch_size,Q_sequence_length,self.multiheads,self.head_dim]
        Q_seq = K.permute_dimensions(Q_seq, pattern=(
        0, 2, 1, 3))  # Q_seq.shape=[batch_size,self.multiheads,Q_sequence_length,self.head_dim]
        # 对K做线性变换，和Q一样
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, shape=(-1, K.shape(K_seq)[1], self.multiheads, self.head_dim))
        K_seq = K.permute_dimensions(K_seq, pattern=(0, 2, 1, 3))
        # 对V做线性变换，和Q一样
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, shape=(-1, K.shape(V_seq)[1], self.multiheads, self.head_dim))
        V_seq = K.permute_dimensions(V_seq, pattern=(0, 2, 1, 3))
        # 计算内积
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / K.sqrt(K.cast(self.head_dim,
                                                                   dtype='float32'))  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.permute_dimensions(A, pattern=(
        0, 3, 2, 1))  # A.shape=[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        # Mask部分：
        # 1.Sequence-wise Mask(axis=1)：这部分不是Attention论文提出的操作，而是常规应该有的mask操作（类似于Keras.pad_sequence）
        # 原始输入A的形状，[batch_size,K_sequence_length,Q_sequence_length,self.multiheads]
        # 这部分是为了mask掉sequence的填充部分，比如V_len=5,那么对于A需要在K_sequence_length部分进行mask
        # 这部分不好理解的话可以想象为在句子长度上进行mask，统一对齐到V_len
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, pattern=(
        0, 3, 2, 1))  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # 2.Position-wise Mask(axis=2)：这部分是Attention论文提出的操作，在Encoder时不使用，在Decoder时使用
        # 原始输入A的形状，[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # 这部分是为了mask掉后续Position的影响，确保Position_i的预测输出仅受Position_0~Position_i的影响
        # 这部分不好理解的话可以想象为为进行实时机器翻译时，机器是无法获取到人后面要说的是什么话，它能获得的信息只能是出现过的词语
        if self.mask_right:
            ones = K.ones_like(A[:1, :1])  # ones.shape=[1,1,Q_sequence_length,K_sequence_length],生成全1矩阵
            lower_triangular = K.tf.matrix_band_part(ones, num_lower=-1,
                                                     num_upper=0)  # lower_triangular.shape=ones.shape，生成下三角阵
            mask = (
                               ones - lower_triangular) * 1e12  # mask.shape=ones.shape，生成类上三角阵（注：这里不能用K.tf.matrix_band_part直接生成上三角阵，因为对角线元素需要丢弃），同样需要乘以一个很大的数（减去这个数）,以便在softmax时趋于0
            A = A - mask  # Element-wise subtract，A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        A = K.softmax(A)  # A.shape=[batch_size,self.multiheads,Q_sequence_length,K_sequence_length]
        # V_seq.shape=[batch_size,V_sequence_length,V_embedding_dim]
        O_seq = K.batch_dot(A, V_seq,
                            axes=[3, 2])  # O_seq.shape=[batch_size,self.multiheads,Q_sequence_length,V_sequence_length]
        O_seq = K.permute_dimensions(O_seq, pattern=(
        0, 2, 1, 3))  # O_seq.shape=[batch_size,Q_sequence_length,self.multiheads,V_sequence_length]
        # 这里有个坑，维度计算时要注意：(batch_size*V_sequence_length)/self.head_dim要为整数
        O_seq = K.reshape(O_seq, shape=(
        -1, K.shape(O_seq)[1], self.output_dim))  # O_seq.shape=[,Q_sequence_length,self.multiheads*self.head_dim]
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

class PositionEmbedding(Layer):

    def __init__(self, method='sum', embedding_dim=None, **kwargs):
        """
        # 此层Layer仅可放在Embedding之后。
        # 参数：
        #    - embedding_dim: position_embedding的维度，为None或者偶数（Google给的PositionEmbedding构造公式分奇偶数）；
        #    - method: word_embedding与position_embedding的结合方法，求和sum或拼接concatenate；
        #        -- sum: position_embedding的值与word_embedding相加，需要将embedding_dim定义得和word_embedding一样；默认方式，FaceBook的论文和Google论文中用的都是后者；
        #        -- concatenate：将position_embedding的值拼接在word_embedding后面。
        """
        self.method = method
        self.embedding_dim = embedding_dim
        super(PositionEmbedding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.method == 'sum':
            return input_shape
        elif self.method == 'concatenate':
            return (input_shape[0], input_shape[1], input_shape[2] + self.embedding_dim)
        else:
            raise TypeError('Method not understood:', self.method)

    def call(self, word_embeddings):
        """
        # 参照keras.engine.base_layer的call方法。
        # 将word_embeddings中的第p个词语映射为一个d_pos维的position_embedding，其中第i个元素的值为PE_i(p)，计算公式如下：
        #     PE_2i(p) = sin(p/10000^(2i/d_pos))
        #     PE_2i+1(p) = cos(p/10000^(2i/d_pos))
        # 参数
        #     - word_embeddings: Tensor or list/tuple of tensors.
        # 返回
        #     - position_embeddings：Tensor or list/tuple of tensors.
        """
        if (self.embedding_dim == None) or (self.method == 'sum'):
            self.embedding_dim = int(word_embeddings.shape[-1])
        batch_size, sequence_length = K.shape(word_embeddings)[0], K.shape(word_embeddings)[1]
        # 生成(self.embedding_dim,)向量：1/(10000^(2*[0,1,2,...,self.embedding_dim-1]/self.embedding_dim))，对应公式中的1/10000^(2i/d_pos)
        embedding_wise_pos = 1. / K.pow(10000., 2 * K.arange(self.embedding_dim / 2,
                                                             dtype='float32') / self.embedding_dim)  # n_dims=1, shape=(self.embedding_dim,)
        # 增加维度
        embedding_wise_pos = K.expand_dims(embedding_wise_pos, 0)  # n_dims=2, shape=(1,self.embedding_dim)
        # 生成(batch_size,sequence_length,)向量，基础值为1，首层为0，按层累加(第一层值为0，第二层值为1，...)，对应公式中的p
        word_wise_pos = K.cumsum(K.ones_like(word_embeddings[:, :, 0]),
                                 axis=1) - 1  # n_dims=2, shape=(batch_size,sequence_length)
        # 增加维度
        word_wise_pos = K.expand_dims(word_wise_pos, 2)  # n_dims=3, shape=(batch_size,sequence_length,1)
        # 生成(batch_size,sequence_length,self.embedding_dim)向量，对应公式中的p/10000^(2i/d_pos)
        position_embeddings = K.dot(word_wise_pos, embedding_wise_pos)
        # 直接concatenate无法出现交替现象，应先升维再concatenate再reshape
        position_embeddings = K.reshape(
            K.concatenate([K.cos(position_embeddings), K.sin(position_embeddings)], axis=-1),
            shape=(batch_size, sequence_length, -1))
        if self.method == 'sum':
            return word_embeddings + position_embeddings
        elif self.method == 'concatenate':
            return K.concatenate([word_embeddings, position_embeddings], axis=-1)

class CRF(Layer):
    """纯Keras实现CRF层
    CRF层本质上是一个带训练参数的loss计算层，因此CRF层只用来训练模型，
    而预测则需要另外建立模型。
    """
    def __init__(self, ignore_last_label=False, **kwargs):
        """ignore_last_label：定义要不要忽略最后一个标签，起到mask的效果
        """
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        """递归计算归一化因子
        要点：1、递归计算；2、用logsumexp避免溢出。
        技巧：通过expand_dims来对齐张量。
        """
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        """计算目标路径的相对概率（还没有归一化）
        要点：逐标签得分，加上转移概率得分。
        技巧：用“预测”点乘“目标”的方法抽取出目标路径的得分。
        """
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2 # 两个错位labels，负责从转移矩阵中抽取目标转移得分
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score # 两部分得分之和
    def call(self, inputs): # CRF本身不改变输出，它只是一个loss
        return inputs
    def loss(self, y_true, y_pred): # 目标y_pred需要是one hot形式
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]] # 初始状态
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)
        

        
def get_embedding(model_path):
    #加载模型
    model = gensim.models.Word2Vec.load(model_path)
    #获取词向量和词索引
    word_vector = {}
    word_index = {}
    for word,vector,idx in zip(model.wv.index2word,model.wv.vectors,range(len(model.wv.vectors))):
        word_vector[word] = vector
        word_index[word] = idx
    embedding_matrix = model.wv.vectors
    num_words = embedding_matrix.shape[0]
    embedding_dim = embedding_matrix.shape[1]
    return word_vector,word_index,embedding_matrix,num_words,embedding_dim

def get_tokenizer(num_words,corpus_path):
    logger = logging.getLogger(u'generate tokenizer')
    tokenizer = Tokenizer(num_words=num_words,filters='')
    logger.info(u'loading corpus...')
    with open(corpus_path,'r',encoding='utf-8') as f:
        corpus = f.readlines()
    logger.info(u'normalizing corpus...')
    corpus = [item.replace('\n','') for item in corpus]
    logger.info(u'fit on corpus...')
    tokenizer.fit_on_texts(corpus)
    logger.info(u'done.')
    return tokenizer

def get_text_sequences(tokenizer, texts, maxlen):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=maxlen, padding='post', truncating='post')

def pad_labels(labels, maxlen, pad_item='o', padding='pre', truncating='pre'):
    res = []
    for label in labels:
        if len(label) < maxlen:
            if padding == 'post':
                tmp = label
                tmp.extend([pad_item]*(maxlen-len(label)))
            else:
                tmp = [pad_item]*(maxlen-len(label))
                tmp.extend(label)
        else:
            if truncating == 'post':
                tmp = label[:maxlen]
            else:
                tmp = label[-maxlen:]
        res.append(label)
    return res

def data_normalizer(data,labels):
    global tokenizer
    global tag2id
    global output_classes
    maxlen = max([len(item) for item in data])
    X = get_text_sequences(tokenizer, data, maxlen)
    Y = pad_labels(labels, maxlen, pad_item='o', padding='post', truncating='post')
    Y = [[tag2id[item] for item in term] for term in Y]
    return np.array(X),to_categorical(Y,output_classes)

def k_fold_cross_validation(data,labels,k_fold,model,batch_size,epochs):
    logger = logging.getLogger('training')
    num_data = len(data)
    samples_per_fold = num_data // k_fold
    num_train_data = samples_per_fold * (k_fold - 1)
    num_val_data = samples_per_fold * (k_fold - 1)
    idx = [i for i in range(num_data)]
    np.random.shuffle(idx)
    metrics = {}
    for i in range(k_fold):
        logger.info('training on fold {}'.format(i))
        
        val_data = data[i*samples_per_fold:(i+1)*samples_per_fold]
        val_labels = labels[i*samples_per_fold:(i+1)*samples_per_fold]
        val_data,val_labels = data_normalizer(val_data,val_labels)
        
        train_data = data[:i*samples_per_fold]
        train_data.extend(data[(i+1)*samples_per_fold:])
        train_labels = labels[:i*samples_per_fold]
        train_labels.extend(labels[(i+1)*samples_per_fold:])
        train_data,train_labels = data_normalizer(train_data,train_labels)
        clf_report = ClassificationReport(target_names=tag2id)
        history = model.fit(train_data,train_labels,batch_size=batch_size,validation_data=(val_data,val_labels),epochs=epochs,callbacks=[clf_report])
        train_acc,train_f1,val_acc,val_f1 = history.history['acc'],history.history['f1'],history.history['val_acc'],history.history['val_f1']
        metrics.setdefault('train_acc',[]).append(train_acc)
        metrics.setdefault('train_f1',[]).append(train_f1)
        metrics.setdefault('val_acc',[]).append(val_acc)
        metrics.setdefault('val_f1',[]).append(val_f1)
    metrics['train_acc'] = np.mean(metrics['train_acc'],axis=0)
    metrics['train_f1'] = np.mean(metrics['train_f1'],axis=0)
    metrics['val_acc'] = np.mean(metrics['val_acc'],axis=0)
    metrics['val_f1'] = np.mean(metrics['val_f1'],axis=0)
    return metrics


if __name__ == '__main__':
    #读取词向量
    word2vec_dir = root_dir + 'char2vec/char2vec_datagrand_128dim'
    word_vector,word_index,embedding_matrix,num_words,embedding_dim = get_embedding(word2vec_dir)

    
    #读取数据
    train_path = root_dir + 'data/raw_data/train.txt'
    test_path = root_dir + 'data/raw_data/test.txt'
    with open(train_path,'r',encoding='utf-8') as f:
        train = f.readlines()
    with open(test_path,'r',encoding='utf-8') as f:
        test = f.readlines()
    train = [item.replace('\n','').split('  ') for item in train]
    test = [item.replace('\n','') for item in test]
    
    train_data = []
    train_labels = []
    for term in train:
        train_x = ''
        train_y = ''
        for subitem in term:
            subitem = subitem.split('/')
            seq_ = subitem[0].split('_')
            type_ = subitem[1]
            for i in range(len(seq_)):
                if type_ != 'o':
                    if i == 0:
                        train_y += ('B-' + type_ + '_')
                    elif i == len(seq_)-1:
                        train_y += ('E-' + type_ + '_')
                    else:
                        train_y += ('I-' + type_ + '_')
                else:
                    train_y += (type_ + '_')
                #train_x += (seq_[i] + '|_') #词级别用这个
                train_x += (seq_[i] + '_') #字级别用这个
        train_data.append(train_x[:-1].split('_'))
        train_labels.append(train_y[:-1].split('_'))
        
    #test_data = [item.replace('_','|_') + '|' for item in test] #词级别用这个
    test_data = test #词级别用这个
    
    
    #训练数据处理
    corpus_path = root_dir + 'data/gen_data/corpus_char.txt'
    id2tag = {0:'B-a',1:'I-a',2:'E-a',3:'B-b',4:'I-b',5:'E-b',6:'B-c',7:'I-c',8:'E-c',9:'o'}
    tag2id = {j:i for i,j in id2tag.items()}
    output_classes = len(tag2id)
    tokenizer = get_tokenizer(num_words=num_words, corpus_path=corpus_path) #先生成tokenizer（Keras）
    
    
    #在这里构建你的模型：
    ###注意：
    crf = CRF(True)
    sequence = Input(shape=(None,), dtype='int32')
    embedding = Embedding(input_dim=num_words,output_dim=embedding_dim,weights=[embedding_matrix],trainable=True)(sequence)
    #######----【Encoder部分】从这开始可以插入任意的编码器模型（多分类模型），输出维度一定要是【三维】，可以在本部分的最后的部分print看一下，比如print(output.shape)----#######
    embedding = PositionEmbedding()(embedding)
    attention = Attention(multiheads=8,head_dim=16,mask_right=False)([embedding,embedding,embedding])
    attention_layer_norm = LayerNormalization()(attention)
    dropout = Dropout(0.5)(attention_layer_norm)
    dense = Dense(128, activation='relu')(dropout)
    dense_layer_norm = LayerNormalization()(dense)
    output = dense_layer_norm
    #######----【Decoder部分】从这开始可以插入任意的解码器模型（多分类模型），输出维度一定要是【二维】，可以在本部分的最后的部分print看一下，比如print(output.shape)----#######
    output = TimeDistributed(Dense(output_classes, activation='softmax'))(output)
    output = crf(output)
    #######到这里结束-----------------------------------------------------------------------------------------------------------------------------------------#######
    my_model = Model(input=sequence, output=output)
    my_model.compile(loss=crf.loss, optimizer='adam', metrics=[crf.accuracy, Metrics().f1]) ###如果用了CRF，一定要用CRF本身的loss和metrics
    #my_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc',Metrics().f1]) ###如果用了其他解码器，可以选择多分类的交叉熵loss


    #模型评估
    ###这个metrics不要看，虚高！！要看训练过程中的callback回调！！
    ###下面的参数中，my_model改为上面写的模型的名字
    metrics = k_fold_cross_validation(train_data,train_labels,5,my_model,batch_size=256,epochs=20) 