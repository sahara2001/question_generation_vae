import os
import sys
root_dir = '/home1/liushaoweihua/jupyterlab/datagrand'
sys.path.append(root_dir)
os.chdir(root_dir)
from tokenizer import *
import multiprocessing
import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(message)s')
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

def train(corpus_dir, model_dir, train_params):
    allowed_params = {
        'dimension': 128,
        'replace': False,
        'window': 5,
        'min_count': 1,
        'sg': 1,
        'hs': 1,
        'negative': 0,
        'workers': 12,
        'iter': 20,
        'epochs': 5
    }
    dimension = train_params.get('dimension') or allowed_params.get('dimension')
    replace = train_params.get('replace') or allowed_params.get('replace')
    window = train_params.get('window') or allowed_params.get('window')
    min_count = train_params.get('min_count') or allowed_params.get('min_count')
    sg = train_params.get('sg') or allowed_params.get('sg')
#    hs = train_params.get('hs') or allowed_params.get('hs')
    negative = train_params.get('negative') or allowed_params.get('negative')
    workers = train_params.get('workers') or allowed_params.get('workers')
    iter_ = train_params.get('iter') or allowed_params.get('iter')
    epochs = train_params.get('epochs') or allowed_params.get('epochs')

    logger = logging.getLogger(u'训练词向量')
    model_name = 'char2vec_datagrand_' + str(dimension) + 'dim'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    logger.info(u'读取语料中...')
    corpus = LineSentence(corpus_dir)
    if not os.path.exists(model_dir + '/' + model_name) or replace:
        logger.info(u'训练模型中...')
        model = Word2Vec(sentences=corpus,size=dimension,window=window,min_count=min_count,sg=sg,negative=negative,workers=workers,iter=iter_)
        logger.info(u'保存模型中...')
        model.save(model_dir + '/' + model_name)
        logger.info(u'训练完成！')
    else:
        try:
            logger.info(u'加载预训练模型中...')
            model = load(model_dir + '/' + model_name)
            logger.info(u'读取词向量中...')
            model.build_vocab(sentences=corpus,update=True)
            logger.info(u'增量训练模型中...')
            model.train(sentences=corpus,total_examples=model.corpus_count,epochs=epochs)
            logger.info(u'保存模型中...')
            model.save(model_dir + '/' + model_name)
            logger.info(u'训练完成！')
        except:
            logger.error(u'预训练模型加载失败！')
    
def preprocess(raw_corpus_dir, corpus_dir):
    logger = logging.getLogger(u'处理语料')
    logger.info(u'读取语料中...')
    with open(raw_corpus_dir,'r',encoding='utf-8') as f:
        texts = f.readlines()
    corpus = [item.replace('\n','').replace('_',' ') for item in texts]
    logger.info(u'保存结果中...')
    with open(corpus_dir,'w',encoding='utf-8') as f:
        f.writelines(corpus)
 
if __name__ == '__main__':
    preprocess_, train_ = [True, True]
    if preprocess_:
        raw_corpus_dir = root_dir + '/data/raw_data/corpus.txt'
        corpus_dir = root_dir + '/data/gen_data/corpus_char.txt'
        preprocess(raw_corpus_dir, corpus_dir)
    if train_:
        corpus_dir = root_dir + '/data/gen_data/corpus_char.txt'
        model_dir = root_dir + '/char2vec'
        train_params = {
            'dimension': 128,
            'replace': True,
            'window': 5,
            'min_count': 1,
            'sg': 1,
            'negative': 5,
            'workers': 12,
            'iter': 20,
        }
        train(corpus_dir, model_dir, train_params)
