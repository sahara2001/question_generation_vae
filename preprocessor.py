import codecs
import re
import numpy as np

M_WORD = 60

def get_shi(n=5):
    """get sentence from shi.txt
    @return shi: list of  strings
    """
    s = codecs.open('shi.txt', encoding='utf-8').read()
    # for i in s:
    #     print(i)
    # 通过正则表达式找出所有的五言诗
    #  re.compile(r'foo\(.*?\)')
    # s = re.findall(u'(.{%s}，.{%s}。.)\r\n'%(n,n), s)
    s = re.findall(u'([^]+)。([^]+)。\n', s)
    # for i in s:
    #     print(i)
    shi = []
    for i in s:
    #     print(i)
        for j in i.split(u'。'): # 按句切分
            if j:
                shi.append(j)

    shi = [i[:n] + i[n+1:] for i in shi if len(i) == 2*n+1]
    print(f'length {len(shi)}')
    
    return shi

def pad_question(words,m_word, pad_char):
    """pad question with 3 tuple
    @param words(list of int): words to pad.
    @param m_word(int): max word len
    @pad_char (str): pattern to pad
    @return padded_words(list of str)
    """
    return [i+''.join([pad_char]*(m_word-len(i))) for i in words]
    

def get_question():
    """get question training and test set with no 3-tuple"""

    fi = open('nlpcc-iccpol-2016.kbqa.training-data','r',encoding='utf8')
    fii = open('nlpcc-iccpol-2016.kbqa.testing-data','r',encoding='utf8')

    q=''

    train = []
    countChar = {}
    m_word = 0
    i = 0
    for line in fi:
#         print(f'line: {line}')
        if line[1] == 'q':
#             print(f'line Q: {line}')
            q = line[line.index('\t') + 1:].strip()
            if len(q) > m_word:
                m_word = len(q)
            train.append(q)
#             print(f'filtered Q: {q}')
            for char in q:
                if char not in countChar:
                    countChar[char] = 1
                else:
                    countChar[char] += 1
#         elif line[1] == 't':
#             print(f'line P: {line}')
#             sub = line[line.index('\t') + 1:line.index(' |||')].strip()
#             qNSub = line[line.index(' ||| ') + 5:]
#             pre = qNSub[:qNSub.index(' |||')]
#             print(f'sub:{sub}')
#             print(f'qNSub:{qNSub}')
#             print(f'pre:{pre}')

    test = []
    for line in fii:
#         print(f'line: {line}')
        if line[1] == 'q':
#             print(f'line Q: {line}')
            q = line[line.index('\t') + 1:].strip()
            if len(q) > m_word:
                m_word = len(q)
            test.append(q)
#             print(f'filtered Q: {q}')
            for char in q:
                if char not in countChar:
                    countChar[char] = 1
                else:
                    countChar[char] += 1
            
    
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train))
    with open('test.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test))
        
    # Save
    np.save('words.npy', countChar)
    
    print(m_word)

#     # Load npy dict
#     read_dictionary = np.load('my_file.npy').item()
#     print(read_dictionary['hello']) # displays "world"
            
    
    return m_word

def load_nlpcc(m_word):
    # keras embedding filter char
    pad_char = '0'  #not used in training set
    train = []
    test = []
    with open('data/train.txt', 'r', encoding='utf-8') as f:
        train = [i for i in f.readlines()]
    with open('data/test.txt', 'r', encoding='utf-8') as f:
        test = [i for i in f.readlines()]
        
#     words = np.load('data/words.npy').item()
#     count = len(words)
    
    #use keras laryer for masking
    train = pad_question(train, m_word,pad_char)
    test = pad_question(test, m_word,pad_char)
    return train, test
    

def get_vocab(shi):
    """translate char to id
    @return id2char(dict): project id 2 char
    @return char2id(dict): project char 2 id"""
    
    # 构建字与id的相互映射
    id2char = dict(enumerate(set(''.join(shi))))
    char2id = {j:i for i,j in id2char.items()}
    print(f'length {len(id2char)}')

    
    return id2char, char2id



if __name__ == "__main__":
    get_question()