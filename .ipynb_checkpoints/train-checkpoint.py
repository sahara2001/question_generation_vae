import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import keras
from keras.callbacks import Callback
from model import *
from preprocessor import get_shi, get_vocab, load_nlpcc
import numpy as np

M_WORD = 60  #max number of words

# 回调器，方便在训练过程中输出

class Evaluate(Callback):
    def __init__(self, generator, latent_dim, id2char, n):
        self.log = []
        self.gen = generator
        self.latent_dim = latent_dim
        self.id2char = id2char
        self.n = n
    def on_epoch_end(self, epoch, logs=None):
        self.log.append(gen(self.gen, self.latent_dim, self.id2char, self.n))
        print( ('          %s'%(self.log[-1])) )


if __name__=="__main__":
#     n = 5 # 只抽取五言诗
    n = int(M_WORD / 2)
    latent_dim = 384 # 隐变量维度
    hidden_dim = 384 # 隐层节点数
    
#     shi = get_shi()
    shi, test = load_nlpcc(M_WORD)
    

    id2char, char2id = get_vocab(shi+test)
    embedding_len = len(id2char)
    
    # 诗歌id化
    shi2id = [[char2id[j] for j in i] for i in shi]
    shi2id = np.array(shi2id)
    
    vae,generator = VAEModel(embedding_len, hidden_dim, latent_dim, n)
    
    #vae.load_weights('shi.model')
    
    evaluator = Evaluate(generator, latent_dim, id2char, n)
    
    #tensorboard
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    vae.fit(shi2id,
            shuffle=True,
            epochs=500,
            batch_size=512,
            callbacks=[evaluator, tbCallBack])

    vae.save_weights('shi.model')

    for i in range(20):
        print(gen(generator, latent_dim, id2char,n))