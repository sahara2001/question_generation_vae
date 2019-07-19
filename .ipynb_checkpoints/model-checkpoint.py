import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
from modules import GCNN, sampling, ResCNN


def VAEModel(embedding_len, hidden_dim, latent_dim, n):
    """ create and return a vae model
    @param embedding_len(int): length of word embedding
    @param hidden_dim(int): encoder output length
    @param latent_dim(int): dunno
    """
    
    """
    Result: 
    1. Reuse decoder ResCNN with GCNN: Unstable, nan loss after half at most
    2. Reduce to 5 layer: slow as well
    """
    
    ## Encoder
    input_sentence = Input(shape=(2*n,), dtype='int32')
    input_vec = Embedding(embedding_len, hidden_dim, mask_zero=True)(input_sentence) # id转向量
    h = GCNN(residual=True)(input_vec) # GCNN层
    h = GCNN(residual=True)(h) # GCNN层
    h = GCNN(residual=True)(h) # GCNN层
    h = GCNN(residual=True)(h) # GCNN层
    h = GCNN(residual=True)(h) # GCNN层
    h = GlobalAveragePooling1D()(h) # 池化
    
    # 算均值方差
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # map to gaussian
    z = Lambda(sampling(latent_dim))([z_mean, z_log_var])

    ## Decoder
    # 定义解码层，分开定义是为了后面的重用
    decoder_hidden = Dense(hidden_dim*(2*n))
    decoder_cnn1 = GCNN(residual=True)
    decoder_cnn2 = GCNN(residual=True)
    decoder_cnn3 = GCNN(residual=True)
    decoder_cnn4 = GCNN(residual=True)
    decoder_cnn5 = GCNN(residual=True)
    decoder_dense = Dense(embedding_len, activation='softmax')

    h = decoder_hidden(z)
    h = Reshape((2*n, hidden_dim))(h)
    h = decoder_cnn1(h) #reuse is like feed forward rnn, easier to train
    h = decoder_cnn2(h)
    h = decoder_cnn3(h)
    h = decoder_cnn4(h)
    h = decoder_cnn5(h)
    output = decoder_dense(h)

    # 建立模型
    vae = Model(input_sentence, output)
    
    ## Loss
    # xent_loss是重构loss，kl_loss是KL loss
    xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    
    ## Compile Model
    # add_loss是新增的方法，用于更灵活地添加各种loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    
    ## Generator
    # 重用解码层，构建单独的生成模型
    decoder_input = Input(shape=(latent_dim,))
    _ = decoder_hidden(decoder_input)
    _ = Reshape((2*n, hidden_dim))(_)
    _ = decoder_cnn1(_)
    _ = decoder_cnn2(_)
    _ = decoder_cnn3(_)
    _ = decoder_cnn4(_)
    _ = decoder_cnn5(_)
    _output = decoder_dense(_)
    generator = Model(decoder_input, _output)
    
    return vae, generator

# 利用生成模型随机生成一首诗/问题
def gen(generator, latent_dim, id2char,n):
    r = generator.predict(np.random.randn(1, latent_dim))[0]
    r = r.argmax(axis=1)
    return ''.join([id2char[i] for i in r])

    