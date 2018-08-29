# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import config
from utils import load_data, BM25, cal_simulation
from gensim.models import KeyedVectors
from sklearn.externals import joblib
lr = joblib.load(config.stacking_model)

def init_w2v(EMBEDDING_FILE = config.embedding_file):
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False)
    #64维的词向量文件为binary类型
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    return word2vec

#init parameters
word2vec = init_w2v()
stopwords = load_data.stopwordslist(config.stopwords_file)
embedding_size = config.embedding_size
max_len = config.max_len
top_k = config.top_k

#初始化w2v,bm25模型以及加载FAQ库
FAQs = load_data.csv_to_list(config.faq_file,FAQ = "private")
FAQ_vec = cal_simulation.init_cal_sim(FAQs, word2vec, stopwords, max_len)
FAQ_test = load_data.csv_to_list(config.faq_file_test,FAQ = "public")
bm25Model, average_idf, index_faq_bm25 = BM25.init_BM25(FAQs, stopwords, max_len)

correct_num_vector = 0
correct_num_tfidf = 0
correct_num_bm25 = 0
correct_num_combine = 0
average_vector_method = []
average_vector_score = []
s_len_lst = []
miss_word_lst = []
bm25_method = []
bm25_score = []
label = []
bm25_right = []
stacking_right = []
#对测试集每一句开始测试
for faq_test in FAQ_test:
    x_eval_vec, seq_len_eval, miss_word, x_seg, sentence_len = cal_simulation.process_sentence(faq_test.question, word2vec,
                                                                                               stopwords, max_len)
    #得到stacking输出结果
    sorted_result = cal_simulation.cal_sim(faq_test.question, FAQs, FAQ_vec,bm25Model,average_idf,word2vec, stopwords,
                                        embedding_size,query_state='1',max_len=max_len,model = lr)
    #tf-idf输出结果
    # index_and_score_tfidf = tfidf.cal_tfidf_sim(faq_test.question, dictionary, tfidf_vectors, stopwords, max_len, handle,
    #                                             index_faq)
    # index_and_score_tfidf = tfidf.reshape(FAQs, index_and_score_tfidf)
    #BM25输出结果
    index_and_score_bm25 = BM25.cal_BM25_sim(faq_test.question, bm25Model, average_idf, index_faq_bm25, stopwords, max_len)
    index_and_score_bm25 = BM25.reshape(FAQs, index_and_score_bm25)

    #统计每一条测试数据的标签和分数信息
    # if faq_test.business_module == sorted_result[0][-1]:
    #     average_vector_method.append(1)
    #     average_vector_score.append(sorted_result[0][2])
    #     miss_word_lst.append(miss_word)
    #     s_len_lst.append(sentence_len)
    # else:
    #     average_vector_method.append(0)
    #     average_vector_score.append(sorted_result[0][2])
    #     miss_word_lst.append(miss_word)
    #     s_len_lst.append(sentence_len)
    # if faq_test.business_module == index_and_score_bm25[0][-1]:
    #     bm25_method.append(1)
    #     bm25_score.append(index_and_score_bm25[0][2])
    # else:
    #     bm25_method.append(0)
    #     bm25_score.append(index_and_score_bm25[0][2])

    #前top_k项是否含有答案项
    flag_stacking = 0
    flag_bm25 = 0
    for i in range(top_k):
        if faq_test.business_module == sorted_result[i][-1]:
            flag_stacking = 1
            correct_num_combine += 1
            stacking_right.append(1)
        # if faq_test.business_module == sorted_faq[i][-1]:
        #     correct_num_vector += 1
        # if faq_test.business_module == index_and_score_tfidf[i][-1]:
        #     correct_num_tfidf += 1
        # #将top_k设为200，从而得到正负样本
        if faq_test.business_module == index_and_score_bm25[i][-1]:
            flag_bm25 = 1
            correct_num_bm25 += 1
            bm25_right.append(1)
            # # positive sample
            # average_vector_score.append(sorted_result[i][2])
            # bm25_score.append(index_and_score_bm25[i][2])
            # # miss_word_lst.append(miss_word)
            # # s_len_lst.append(sentence_len)
            # label.append(1)
            # #negative sample
            # average_vector_score.append(sorted_result[i+1][2])
            # bm25_score.append(index_and_score_bm25[i+1][2])
            # # miss_word_lst.append(miss_word)
            # # s_len_lst.append(sentence_len)
            # label.append(0)
    if flag_bm25 is 0:
        bm25_right.append(0)
    if flag_stacking is 0:
        stacking_right.append(0)


print("Total test num is {}".format(len(FAQ_test)))
print("correct num of stacking，bm25 are {} and {}".format(correct_num_combine,correct_num_bm25))
print("Top {} Accuracy rate of stacking is {} and {}".format(top_k,float(correct_num_combine) / len(FAQ_test),float(correct_num_bm25) / len(FAQ_test)))

print(stacking_right)
print(bm25_right)
import numpy as np
diff = np.array(stacking_right) - np.array(bm25_right)
diff_question = []
for i in range(len(diff)):
    if diff[i] != 0:
        diff_question.append(FAQ_test[i].question)
print(diff_question)

wrong_question = []

for i in range(len(stacking_right)):
    if stacking_right[i] == 0:
        wrong_question.append((FAQ_test[i].question,FAQ_test[i].business_module))
print(wrong_question)
# print("average vector method prediction are:")
# print(average_vector_method)
# print("######################################")
# print("bm25 method prediction are:")
# print(bm25_method)
# print(average_vector_score)
# print(miss_word_lst)
# print(s_len_lst)


# # 生成stacking所需的数据集
# import pandas as pd
# df = pd.DataFrame(columns=('average_vector_score', 'bm25_score', 'label'))
# for i in range(len(label)):
#     df.loc[i] = [average_vector_score[i],bm25_score[i],label[i]]
# df.to_csv("stacking.csv")