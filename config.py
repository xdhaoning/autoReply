# -*- coding: utf-8 -*-
#配置文件

#数据目录
data_dir = "./data/"

#词向量bin文件
#50维词向量
embedding_file = data_dir + "zhwiki_2017_03.sg_50d.word2vec"
embedding_size = 50
#64维词向量
# embedding_file = data_dir + "news_12g_baidubaike_20g_novel_90g_embedding_64.bin"
# embedding_size = 64

#50维词向量的分类器模型
classifier_model_ckpt = data_dir + "model_50/"
stacking_model = data_dir + "train_model.m"
image = data_dir + "python1.gif"
#64维词向量的分类器模型
# classifier_model_ckpt = data_dir + "model_64/"

#FAQ库文件
faq_file = data_dir + "FAQ_with_sim.csv"
faq_file_common = data_dir + "public_FAQ.csv"
faq_file_test = data_dir + "test_FAQ_new_50.csv"
#停用词路径
stopwords_file = data_dir + 'stop_words.txt'
#返回top_k个数
top_k = 5
#max length of sentence
max_len = 20
#是否打开闲聊机器人
chatBotFlag = False
#是否打开业务非业务分类(和闲聊机器人绑定打开)
classify_flag = False
#是否需要查询公共库
hasPublicFAQ = False

#stacking file
stacking_data = data_dir + "stacking.csv"