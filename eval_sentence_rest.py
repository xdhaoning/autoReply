#coding=utf-8
import json
import numpy as np
import tensorflow as tf
import warnings
import config
from utils import load_data, cal_simulation, BM25, chatBot, py2json
from sklearn.externals import joblib
warnings.filterwarnings('ignore')
#载入stacking模型
lr = joblib.load(config.stacking_model)

#初始化业务分类模型
def init_classifier(MODEL_FILE = config.classifier_model_ckpt):
    saver=tf.train.import_meta_graph(MODEL_FILE + 'model.ckpt.meta')
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_FILE))
    predict = sess.graph.get_tensor_by_name('output:0')
    x = sess.graph.get_tensor_by_name('x:0')
    seq_len = sess.graph.get_tensor_by_name('seq_len:0')
    return sess,predict,x,seq_len
#分类器对句子进行分类
def classify_sentence(x_eval_vec,seq_len_eval,sess,predict,x):
    prediction = sess.run(predict,feed_dict={x:x_eval_vec, seq_len: seq_len_eval})
    return np.argmin(prediction)

#初始化参数
classify_flag = config.classify_flag
stopwords = load_data.stopwordslist(config.stopwords_file)
word2vec = load_data.load_word2vec(config.embedding_file)
embedding_size = config.embedding_size
max_len = config.max_len
top_k = config.top_k
if classify_flag:
    sess, predict, x, seq_len = init_classifier()

#初始化神经网络配置以及加载FAQ库
FAQs = load_data.csv_to_list(config.faq_file,FAQ = "private")
FAQ_vec = cal_simulation.init_cal_sim(FAQs, word2vec, stopwords, max_len)
FAQs_common = load_data.csv_to_list(config.faq_file_common,FAQ = "public")
FAQ_vec_common = cal_simulation.init_cal_sim(FAQs_common, word2vec, stopwords, max_len)
bm25Model, average_idf, index_faq_bm25 = BM25.init_BM25(FAQs, stopwords, max_len)
bm25Model_common, average_idf_common, index_faq_bm25_common = BM25.init_BM25(FAQs_common, stopwords, max_len)

#初始化chatBot
app_key = 'uvwI6m4vxgxt2Mae'
app_id = '1106977811'
chatBotFlag = False
hasPublicFAQ = config.hasPublicFAQ

from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/autoreply', methods=['POST'])
def autoprint():
    # 需要从request对象读取表单内容：
    data = str(request.get_data(),encoding='utf-8')
    sentence = eval(data)['userquestion']['content']
    x_eval_vec, seq_len_eval, miss_word, x_seg, sentence_len = cal_simulation.process_sentence(sentence, word2vec, stopwords, max_len)
    if classify_flag:
        result = classify_sentence(x_eval_vec,seq_len_eval,sess,predict,x)
    else:
        result = 1

    if result == 1:
        # 利用平均词向量法构成句子向量
        sorted_faq = cal_simulation.cal_sim(sentence, FAQs, FAQ_vec, bm25Model, average_idf, word2vec,
                                            stopwords, embedding_size, query_state='1', max_len=max_len, model=lr)
        max_score = sorted_faq[0][2]
        # 若在私有库中查不到，则在公共库中查询。
        if max_score < 0.6 and hasPublicFAQ :
            sorted_faq_common = cal_simulation.cal_sim(sentence, FAQs_common, FAQ_vec_common, bm25Model, average_idf,
                                                       word2vec,stopwords, embedding_size, query_state='1', max_len=max_len,model=lr, public=True)
            if sorted_faq_common[0][2] > 0.8:
                sorted_faq = sorted(sorted_faq_common + sorted_faq, key=lambda x: x[2], reverse=True)
        reply = cal_simulation.get_item(sorted_faq, top_k=top_k)
        if reply[0] is -1:
            responsetype = 3
            reply = ["不好意思，我还不懂您的问题"]
        elif len(reply[0]) > 1:
            responsetype = 2
        else:
            responsetype = 1
    elif chatBotFlag is True:
        responsetype = 8
        ai_obj = chatBot.AiPlat(app_id, app_key)
        rsp = ai_obj.getNlpTextTrans(sentence)
        if rsp['ret'] == 0:
            reply = rsp["data"]["answer"]
        else:
            reply = ["你说的什么什么啊？"]
    else:
        responsetype = 8
        reply = ["请输入业务相关问题"]
    return json.dumps(py2json.reply2json(reply,responsetype),ensure_ascii=False,indent=4)
    # return json.dumps(py2json.reply2json(reply, responsetype))
if __name__ == '__main__':
    app.run(port = 5000,host="0.0.0.0")
    # app.run()
