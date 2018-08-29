# -*- coding:utf-8 -*-
from gensim.summarization import bm25
from utils import load_data
def init_BM25(FAQs,stopwords,max_len,handle = None):
    corpus = []
    index_faq = []
    count = 0
    for faq in FAQs:
        question = load_data.process_sentence(faq.question, stopwords, max_len, handle)[0]
        corpus.append(question)
        index_faq.append((count,0,faq.business_module))
        if len(faq.sim_question) > 0:
            i = 1
            for s in faq.sim_question:
                s = load_data.process_sentence(s, stopwords, max_len, handle)[0]
                corpus.append(s)
                #indicate the faq_id , sim_question_id   and  idx
                index_faq.append((count,i,faq.business_module))
                i += 1
        count += 1
    bm25Model = bm25.BM25(corpus)
    average_idf = sum(map(lambda k: float(bm25Model.idf[k]), bm25Model.idf.keys())) / len(bm25Model.idf.keys())
    return bm25Model,average_idf,index_faq

def cal_BM25_sim(sentence,bm25Model,average_idf,index_faq,stopwords,max_len,handle = None):
    query = load_data.process_sentence(sentence, stopwords, max_len, handle)[0]
    scores = bm25Model.get_scores(query, average_idf)
    tmp = list(zip(scores,index_faq))
    #sorted the faq with BM25 score
    index_and_score = sorted(tmp,key=lambda x : x[0],reverse=True)
    return index_and_score

def reshape(FAQs,index_and_score):
    reshape_score = []
    idx_list = []
    for i in range(len(index_and_score)):
        index = index_and_score[i][1][0]
        sim_id = index_and_score[i][1][1]
        idx = index_and_score[i][1][2]
        score = index_and_score[i][0]
        #deleti the dump item
        if idx not in idx_list:
            idx_list.append(idx)
            if sim_id > 0:
                match_question = FAQs[index].sim_question[sim_id - 1]
            else:
                match_question = FAQs[index].question
            ans = FAQs[index].ans
            reshape_score.append((match_question,ans,score,idx))
    return reshape_score

def print_top_k(index_and_score,top_k):
    for i in range(top_k):
        score = index_and_score[i][2]
        question = index_and_score[i][0]
        ans = index_and_score[i][1]
        if score > 0.8:
            print(question)
            print(ans)
            print(score)


if __name__ == '__main__' :
    import config
    FAQs = load_data.csv_to_list("./data/FAQ_with_sim.csv", FAQ = "private")
    stopwords = load_data.stopwordslist(config.stopwords_file)
    max_len = config.max_len
    handle = load_data.init_seg()

    bm25Model, average_idf, index_faq = init_BM25(FAQs,stopwords,max_len,handle)
    sentence = "员工入职当天需要填写什么材料？"
    index_and_score = cal_BM25_sim(sentence, bm25Model, average_idf, index_faq, stopwords, max_len, handle)
    index_and_score = reshape(FAQs,index_and_score)
    print_top_k(index_and_score,3)