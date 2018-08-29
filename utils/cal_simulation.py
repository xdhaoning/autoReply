# -*- coding:utf-8 -*-
import jieba

from utils import load_data, word_embed, vector
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')


def process_sentence(sentence,word2vec,stopwords,add_dim = True,n_steps = 20,handle = None):
    x_eval,seq_len_eval = load_data.process_sentence(sentence,stopwords,n_steps,handle)
    x_seg = x_eval
    seq_len = seq_len_eval
    #转为向量(需要对x和len增加维度，才能喂进feed_dict)
    seq_len_eval = np.array(seq_len_eval)
    x_eval,miss_word = word_embed.embed_sentence(x_eval, word2vec, n_steps)
    x_eval_vec = np.array(x_eval)
    if add_dim:
        seq_len_eval = seq_len_eval[np.newaxis]
        x_eval_vec = x_eval_vec[np.newaxis, :]
    return x_eval_vec,seq_len_eval,miss_word, x_seg, seq_len

#将单个faq中的问题答案相似问法转为平均向量
def faq_to_average_vec(faq,stopwords,word2vec,handle,max_len = 20):
    f = copy.deepcopy(faq)
    question_seg,question_len = sentence2seg(faq.question,stopwords,handle,max_len)
    f.question= vector.average_vector((word_embed.embed_sentence(question_seg, word2vec,max_len))[0],question_len)
    if len(faq.sim_question) > 0:
        for i in range(len(faq.sim_question)):
            sim_question_seg,sim_question_len = sentence2seg(faq.sim_question[i],stopwords,handle,max_len)
            f.sim_question[i] = vector.average_vector(word_embed.embed_sentence(sim_question_seg, word2vec,max_len)[0],sim_question_len)
    return f


#分词，去停用词处理
def sentence2seg(sentence,stopwords,handle,max_len = 20):
    string = load_data.delete_symbol(sentence)
    s = " ".join(jieba.cut(string.strip())).split()
    # s = (call_TCseg.call_seg(string.encode('gbk'), handle)).decode('gbk').split()
    seq = []
    seq_len = 0
    for word in s:
        if word not in stopwords:
            if word != '\t':
                seq.append(word)
                seq_len += 1
    if len(seq) > max_len:
        return seq[:max_len],max_len
    else:
        return seq,seq_len


def faq_to_weighting_average_vec(faq, faq_tfidf, stopwords, word2vec, max_len, handle):
    f = copy.deepcopy(faq)
    question_seg, question_len = sentence2seg(faq.question, stopwords, handle, max_len)
    f.question = vector.weighting_average_vector((word_embed.embed_sentence(question_seg, word2vec, max_len))[0],faq_tfidf.question,question_len)
    if len(faq.sim_question) > 0:
        for i in range(len(faq.sim_question)):
            sim_question_seg,sim_question_len = sentence2seg(faq.sim_question[i],stopwords,handle,max_len)
            f.sim_question[i] = vector.weighting_average_vector(word_embed.embed_sentence(sim_question_seg, word2vec,max_len)[0],
                                                                faq_tfidf.sim_question[i],sim_question_len)
    return f

def init_cal_sim(FAQs,word2vec,stopwords,max_len,handle = None):
    FAQ_vec = []
    for i in range(len(FAQs)):
        FAQ_vec.append(faq_to_average_vec(FAQs[i], stopwords, word2vec,max_len=max_len,handle = handle))
        # FAQ_vec.append(faq_to_weighting_average_vec(FAQs[i], FAQs_tfidf[i],stopwords, word2vec, max_len=max_len, handle=handle))
    return FAQ_vec

def cal_sim(sentence,FAQs,FAQ_vec,bm25Model,average_idf,word2vec,stopwords,embedding_size,query_state = '1',max_len = 20,handle = None,model = None,public = False):
    x_eval_vec, seq_len_eval,_miss_word,seg_word, _seq_len = process_sentence(sentence, word2vec, stopwords, add_dim=False,n_steps = max_len, handle = handle)
    bm25_scores = bm25Model.get_scores(seg_word, average_idf)
    set_query = set(seg_word)
    # query = load_data.Query(vector.weighting_average_vector(x_eval_vec, tfidf[dictionary.doc2bow(_seg_word)], seq_len_eval))
    query = load_data.Query(vector.average_vector(x_eval_vec,  seq_len_eval))
    query.state = query_state
    question_and_score = []
    index = 0
    for f in range(len(FAQ_vec)):
        faq = FAQ_vec[f]
        #判断是否符合职位，区域信息
        if (query.region == '1' or query.region in faq.region) and (query.state == '1' or query.state in faq.state):
            w2v_score = vector.cosine_distance(query.query,faq.question,embedding_size)
            if w2v_score >= 1:  #此时完全匹配
                final_score = w2v_score
            else:
            #calculate the final score by linear model
            # final_score = model.predict(np.array([w2v_score,bm25_scores[index]]).reshape(1,-1))
            #logistic regression with score parameters
                if public:
                    final_score = w2v_score
                else:
                    # final_score = model.predict_proba(np.array([w2v_score, bm25_scores[index]]).reshape(1, -1))[0][1]
                    # #logistic regression with muti parameters
                    question_seg,question_len = load_data.process_sentence(FAQs[f].question,stopwords,max_len)
                    set_question = set(question_seg)
                    hit_num = len(set_question & set_query)
                    final_score = model.predict_proba(np.array([w2v_score, bm25_scores[index],_miss_word,_seq_len,hit_num,hit_num/_seq_len]).reshape(1, -1))[0][1]

            index += 1
            #查找相似问法中是否有相似的句子,取相似问法最大得分
            if len(faq.sim_question) > 0:
                match_question = FAQs[faq.idx].question
                match_ans = FAQs[faq.idx].ans
                for i in range(len(faq.sim_question)):
                    sim_w2v_score = vector.cosine_distance(query.query,faq.sim_question[i],embedding_size)
                    #ridge regression
                    # sim_final_score = model.predict(np.array([sim_w2v_score, bm25_scores[index + i]]).reshape(1, -1))
                    #logistic regression
                    if sim_w2v_score > 0.99:
                        sim_final_score = sim_w2v_score
                    else:
                        if public:
                            sim_final_score = sim_w2v_score
                        else:
                            # sim_final_score = model.predict_proba(np.array([sim_w2v_score, bm25_scores[index + i]]).reshape(1, -1))[0][1]
                            #logistic regression with multi parameters
                            sim_question_seg, sim_question_len = load_data.process_sentence(FAQs[f].sim_question[i], stopwords, max_len)
                            set_sim_question = set(sim_question_seg)
                            hit_num_sim = len(set_sim_question & set_query)
                            sim_final_score = model.predict_proba(np.array([sim_w2v_score, bm25_scores[index + i],_miss_word,_seq_len,hit_num_sim,hit_num_sim/_seq_len]).reshape(1, -1))[0][1]
                    if sim_final_score > final_score:
                        final_score = sim_final_score
                        match_ans = FAQs[faq.idx].ans
                        match_question = FAQs[faq.idx].sim_question[i]
                index += len(faq.sim_question)
                question_and_score.append((match_question,match_ans,final_score,faq.business_module))
            else:
                question_and_score.append((FAQs[faq.idx].question,FAQs[faq.idx].ans,final_score,faq.business_module))
    question_and_score = sorted(question_and_score,key=lambda x : x[2],reverse=True)
    return question_and_score


def print_item(question_and_score, top_k):
    if question_and_score[0][2] > 0.95:
        # 直接输出
        print(question_and_score[0][0])
        print(question_and_score[0][1])
        print(question_and_score[0][2])
    elif question_and_score[0][2] < 0.4:
        # 无相似问题
        print("找不到符合您当前职位或区域的问题")
    else:
        # 返回top_k
        for i in range(top_k):
            if i < len(question_and_score):
                print(question_and_score[i][0])
                print(question_and_score[i][1])
                print(question_and_score[i][2])

def get_item(question_and_score, top_k):
    question_list = []
    ans_list = []
    score_list = []
    module_list = []
    if question_and_score[0][2] >0.95:
        # 直接输出
        question_list.append(question_and_score[0][0])
        ans_list.append(question_and_score[0][1])
        score_list.append(question_and_score[0][2])
        module_list.append(question_and_score[0][3])
        #print (q1,q2,q3)
        return question_list,ans_list,score_list,module_list

    elif question_and_score[0][2] < 0.4:
        # 无相似问题
        return -1,-1,-1,-1
    else:
        # 返回top_k
        for i in range(top_k):
            if i < len(question_and_score):
                question_list.append(question_and_score[i][0])
                ans_list.append(question_and_score[i][1])
                score_list.append(question_and_score[i][2])
                module_list.append(question_and_score[i][3])
        return question_list,ans_list,score_list,module_list