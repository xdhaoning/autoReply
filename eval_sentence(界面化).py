#coding=utf-8
import numpy as np
import tensorflow as tf
import warnings
import config
from utils import load_data, cal_simulation, BM25, chatBot
from gensim.models import KeyedVectors
from sklearn.externals import joblib
from tkinter import *
import time

warnings.filterwarnings('ignore')
#载入stacking模型
lr = joblib.load(config.stacking_model)

#初始化业务分类模型
def init_classifier(EMBEDDING_FILE = config.embedding_file,MODEL_FILE = config.classifier_model_ckpt):
    word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=False)
    #64维的词向量文件为binary类型
    # word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    saver=tf.train.import_meta_graph(MODEL_FILE + 'model.ckpt.meta')
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(MODEL_FILE))
    predict = sess.graph.get_tensor_by_name('output:0')
    x = sess.graph.get_tensor_by_name('x:0')
    seq_len = sess.graph.get_tensor_by_name('seq_len:0')
    return sess,word2vec,predict,x,seq_len
#分类器对句子进行分类
def classify_sentence(x_eval_vec,seq_len_eval,sess,predict,x):
    prediction = sess.run(predict,feed_dict={x:x_eval_vec, seq_len: seq_len_eval})
    return np.argmin(prediction)

#初始化参数
sess,word2vec,predict,x,seq_len = init_classifier()
stopwords = load_data.stopwordslist(config.stopwords_file)
embedding_size = config.embedding_size
max_len = config.max_len
top_k = config.top_k

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
# chatBotFlag = config.chatBotFlag
chatBotFlag = True

def cmd_mode():
    print("%%%%%%%%%%%%%%%%%%输入要分类的文本，Enter确认%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%输入#结束%%%%%%%%%%%%%%%%%%%")
    sentence = input()
    while sentence != '#' :
        x_eval_vec, seq_len_eval, miss_word, x_seg, sentence_len = cal_simulation.process_sentence(sentence, word2vec, stopwords, max_len)
        result = classify_sentence(x_eval_vec,seq_len_eval,sess,predict,x)
        print("Is business question: {}".format(result))
        print("%%%%%%%%%%%%%%%%%%Segment Result%%%%%%%%%%%%%%%%%%")
        print(" ".join(x_seg))
        print("Miss word num is {}".format(miss_word))

        if result == 1:
            # 利用平均词向量法构成句子向量
            sorted_faq = cal_simulation.cal_sim(sentence, FAQs, FAQ_vec, bm25Model, average_idf, word2vec,
                                                       stopwords,embedding_size, query_state='1', max_len=max_len,model=lr)
            max_score = sorted_faq[0][2]
            #若在私有库中查不到，则在公共库中查询。
            if max_score < 0.6:
                sorted_faq_common = cal_simulation.cal_sim(sentence, FAQs_common, FAQ_vec_common, bm25Model_common, average_idf_common, word2vec,
                                                       stopwords,embedding_size, query_state='1', max_len=max_len,model=lr)
                if sorted_faq_common[0][2] > 0.8:
                    sorted_faq = sorted(sorted_faq_common + sorted_faq,key=lambda x : x[2],reverse=True)

            cal_simulation.print_item(sorted_faq, top_k=top_k)
        elif chatBotFlag is True:
            ai_obj = chatBot.AiPlat(app_id, app_key)
            rsp = ai_obj.getNlpTextTrans(sentence)
            if rsp['ret'] == 0:
                print(rsp["data"]["answer"])
            else:
                print("你说的什么什么啊？")
        else:
            print("请输入业务相关问题")
        sentence = input()
    sess.close()

def gui_mode():
    global m, n, reply, result
    reply = []
    def message(reply):
        strMsg1 = '小小:' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        txtMsgList.insert(END, strMsg1)
        for i in range(len(reply)):
            txtMsgList.insert(END, reply[i])
        txtMsgList.insert(END, txtMsg.get('0.0', END))
        txtMsg.delete('0.0', END)
        return

    def sendMsg():  # 发送消息
        # 在聊天内容上方加一行 显示发送人及发送时间
        strMsg = '我:' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n '
        txtMsgList.insert(END, strMsg)
        txtMsgList.insert(END, txtMsg.get('0.0', END))
        sentence = txtMsg.get('0.0', END)  # 获取输入的内容
        txtMsg.delete('0.0', END)
        # 分类句子是否业务问题
        x_eval_vec, seq_len_eval, miss_word, x_seg, sentence_len = cal_simulation.process_sentence(sentence, word2vec, stopwords, max_len)
        global result
        result = classify_sentence(x_eval_vec, seq_len_eval, sess, predict, x)
        print(result)
        # result = 1
        global reply
        if result == 1:
            sorted_faq = cal_simulation.cal_sim(sentence, FAQs, FAQ_vec, bm25Model, average_idf, word2vec,
                                                stopwords, embedding_size, query_state='1', max_len=max_len, model=lr)
            # 若匹配到的信息不符合，则在公共库中查找
            if sorted_faq[0][2] < 0.6:
                sorted_faq_common = cal_simulation.cal_sim(sentence, FAQs_common, FAQ_vec_common, bm25Model_common,
                                                           average_idf_common, word2vec,
                                                           stopwords, embedding_size, query_state='1', max_len=max_len,
                                                           model=lr,public=True)
                sorted_faq = sorted(sorted_faq_common + sorted_faq, key=lambda x: x[2], reverse=True)

            question_list, ans_list,score_list,module_list = cal_simulation.get_item(sorted_faq, top_k=top_k)
            # 根据第一项的分数决定回复的方式
            if question_list is -1:
                reply = ["找不到符合您当前职位或区域的问题"]
            else:
                reply=[[],[]]

                for i in range(len(question_list)):
                    reply[0].append(question_list[i])
                    reply[1].append(ans_list[i])
                print(reply[0]) #搜到的相似问题
                print(reply[1])
        elif chatBotFlag is True:
            ai_obj = chatBot.AiPlat(app_id, app_key)
            rsp = ai_obj.getNlpTextTrans(sentence)
            if rsp['ret'] == 0:
                reply = [[rsp["data"]["answer"]]]
                print(reply)
            else:
                reply = [["你说的什么啊？"]]
        else:
            reply = [["请输入业务相关问题~"]]
        global m, n
        m = txtMsgList.size()
        print(m)
        message(reply[0])
        n = txtMsgList.size()
        print(n)
        if result == 1:
            if n-m-2 == 3:
                txtMsgList.itemconfig(m-2, fg="green")
                txtMsgList.itemconfig(m, fg="green")
                txtMsgList.itemconfig(m+1, fg="blue", bg="#f0f0ff")
                txtMsgList.itemconfig(m+2, fg="blue", bg="#f0f0ff")
                txtMsgList.itemconfig(m+3, fg="blue", bg="#f0f0ff")
            else:
                txtMsgList.itemconfig(m - 2, fg="green")
                txtMsgList.itemconfig(m, fg="green")
                txtMsgList.itemconfig(m + 1, fg="blue", bg="#f0f0ff")
        else:
            txtMsgList.itemconfig(m-2, fg="green")
            txtMsgList.itemconfig(m, fg="green")
        txtMsgList.see(n)
        # 显示输出

    def cancelMsg():  # 取消消息
        txtMsg.delete('0.0', END)

    def sendMsgEvent(event):  # 发送消息事件
        #if event.keysym == "Return":
        sendMsg()

    def printList(event):
        row = int(txtMsgList.curselection()[0])
        print(row)
        print(reply[1][row-m-1])
        re = reply[1][row-m-1]
        text.set(re)

    # 创建窗口
    t = Tk()
    t.title('智能客服小小与我聊天中')
    t.resizable(0, 0)  # 禁止调整窗口大小

    # 创建frame容器(宽度，高度，背景)
    frmLT = Frame(width=500, height=320, bg='white')
    frmLC = Frame(width=500, height=150, bg='white')
    frmLB = Frame(width=500, height=30)
    frmRT = Frame(width=200, height=500)

    # 创建控件
    bar = Scrollbar(frmLT, width=18)
    bar2 = Scrollbar(frmLT, orient=HORIZONTAL)
    bar2.pack(side=BOTTOM, fill=X)
    bar.pack(side=RIGHT, fill=Y)
    txtMsgList = Listbox(frmLT, height=20)  #连接listbox 到 vertical scrollbar
    bar.config(command=txtMsgList.yview)
    bar2.config(command=txtMsgList.xview)
    txtMsgList.config(width=68, yscrollcommand=bar.set, xscrollcommand=bar2.set)
    txtMsgList.pack(side=LEFT, fill=BOTH)
    txtMsg = Text(frmLC, height=15)

    text = StringVar()
    lb1 = Label(frmRT, text="HR服务", font=("微软雅黑", "13", 'bold'))
    lb1.pack()
    #bm = PhotoImage(file='python2.gif')
    #lb2 = Label(frmRT, image=bm, compound='top')
    #lb2.pack()
    lb = Label(frmRT, height=35, width=50, bg='LightSteelBlue',
               font=("Times", "9", 'bold'),
               justify='left', wraplength=350,
               anchor='nw', textvariable=text)
    lb.pack()

    # 发送消息事件
    txtMsg.bind("<Return>", sendMsgEvent)  # 事件绑定，定义快捷键
    txtMsgList.bind('<Double-Button-1>', printList)
    btnSend = Button(frmLB, text='发送', width=8, command=sendMsg)
    btnCancel = Button(frmLB, text='取消', width=8, command=cancelMsg)

    # 窗口布局(span为跨越数，LT中columnspan(2)意为LT跨越两列，padx/pady意为分割比例为1/3)
    frmLT.grid(row=0, column=0, columnspan=2, padx=1, pady=3)
    frmLC.grid(row=1, column=0, columnspan=2, padx=1, pady=3)
    frmLB.grid(row=2, column=0, columnspan=2)
    frmRT.grid(row=0, column=2, rowspan=3, padx=2, pady=3)

    # 固定大小
    frmLT.grid_propagate(0)
    frmLC.grid_propagate(0)
    frmLB.grid_propagate(0)
    frmRT.grid_propagate(0)

    # 第3行第1列插入按钮Send
    btnSend.grid(row=2, column=0)
    btnCancel.grid(row=2, column=1)
    txtMsg.grid()

    # 主事件循环
    t.mainloop()


if __name__ == '__main__':
    # 界面模式
    gui_mode()

    # 命令行模式
    # cmd_mode()