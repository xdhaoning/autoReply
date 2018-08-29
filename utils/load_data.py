# -*- coding: gbk -*-
"""
load train and test data for training and evaluating process.
"""
import codecs
import re
import jieba
import pandas as pd
from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings('ignore')


# n_steps = 20  # max sentence length
class QApair:
    def __init__(self,question,ans,idx = -1,question_type = 0,region = 1,state = 1,sim_question = [],business_module = 'Unknown',score = 0):
        self.question = question
        self.ans = ans
        self.idx = idx
        self.question_type = question_type
        self.region = region
        self.state = state
        self.sim_question = sim_question
        self.business_module = business_module
        self.score = score
class Query:
    def __init__(self,query,region = '1',state = '1',moduleID = 'unknown'):
        self.query = query
        self.region = region
        self.state = state
        self.moduleID = moduleID


def stopwordslist(filepath):
    stopwords = [line.strip() for line in codecs.open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
def load_word2vec(EMBEDDING_FILE,EMBEDDING_SIZE = 50):
    if EMBEDDING_SIZE is 50:
        word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False)
    else:
        #64维的词向量文件为binary类型
        word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    return word2vec
# 删除过滤掉特殊符号
def delete_symbol(l):
    r1 = u'[!"#$%&\'()*+,-./:;<=>?@，。?★▽￣の、\\…【】《》？（）“”‘’＜ ＞「」：！\\\[\\]^_`{|}~]+'
    return re.sub(r1,'',l.strip())

#处理一个句子(segment)
def process_sentence(sentence,stopwords,n_steps,handle = None,delete_stopwords = True):
    string = delete_symbol(sentence)
    s = " ".join(jieba.cut(string.strip())).split()
    # s = (call_TCseg.call_seg(string.encode('gbk'), handle)).decode('gbk').split()
    if delete_stopwords:
        l = []
        for word in s:
            if word not in stopwords:
                if word != '\t':
                    l.append(word)
        if len(l) > n_steps:
            return l[:n_steps],n_steps
        else:
            return l,len(l)
    else:
        if len(s) > n_steps:
            return s[:n_steps],n_steps
        else:
            return s,len(s)

#读取存储的FAQ库
def csv_to_list(filepath,FAQ):
    df = pd.read_csv(filepath,encoding='utf-8')
    df = df.fillna('')
    FAQs = []
    for i in range(len(df)):
        question = df.iloc[i]['question']
        ans = df.iloc[i]['answer']
        business_module = df.iloc[i]['module']
        idx = i
        question_type = df.iloc[i]['type']
        region = df.iloc[i]['region']
        state = df.iloc[i]['state']
        # 将有相似问法的问法加入到sim_question中
        if FAQ is "private":
            sim_questions = eval(df.iloc[i]['similiar_question'])
        else:
            if len(df.iloc[i]['similiar_question']) > 0:
                sim_questions = df.iloc[i]['similiar_question'].split(' ')
            else:
                sim_questions = []
        sim_tmp = []
        if len(sim_questions) > 0:
            for q in sim_questions:
                sim_tmp.append(q)
        # 实例化
        faq = QApair(question, ans, idx, question_type, region, state, sim_tmp, business_module)
        FAQs.append(faq)
    return FAQs