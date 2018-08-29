依赖库：
	1.anaconda3
	2.tensorflow (若不使用业务非业务问题区分功能，则不需要)
	3.gensim
	4.jieba (或者使用AiLab的分词)


1.词向量使用的为基于wiki百科预训练好的模型，地址为http://pan.baidu.com/s/1dFeNNK9
2.在服务器端运行 eval_sentence_rest.py文件后即可开放5000端口，本地观察效果则运行 eval_sentence(界面化).py， 相关参数配置都放在config.py里。
3.对业务分类的BiLSTM网络模型参考： http://git.code.oa.com/AI_algorithm/QuickReply

关于Rest服务的数据格式：
示例如下，5000端口接收到json格式的请求后，给出json格式的返回
request（请求）
{"businessid":"HRIS","sessionid":"HRIS_yapingwang_20171207145557593","userprofile":{"rtx_ename":"yapingwang","rtx_name":"王亚平","worksitecode":"深圳","emptypecode":"正式","managelevel":"","majorlevel":"","issecretary":"N"},"userquestion":{"content":"社保"}}

response（返回）
{"RtxSessionInfoId":3055,"Code":0,"Msg":"OK","BusinessId":"HRIS","SessionId":"HRIS_yapingwang_20171207145557593","ResponseType":2,"ListResAnswerList":null,"ListResAnswer":{"RtxAnswerInfoId":0,"Cnt":5,"HuaShu":"为您找到以下相关问题，点击可获取对应答案：","Flag_HuaShu":"1","MoreHuaShu":null,"MoreTag":null,"IsSeficiencyFaq":null,"Area":null,"AnswerList":[{"RtxAnswerInfo_ResId":0,"Score":0.64130109004241775,"Rank":"1","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0603","FaqCode":"hris_CB-0603","FaqStaffType":"[正式]","FaqPlace":"[深圳]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d5fd83b0532f0611b6d4172854aa","FaqHot":"49","FaqTitle":"公司和个人的社保缴交比例各是多少？","FaqAnswer":"<p>你好，请参考：</p>\n\n<div>\n<div>深户公司缴交养老保险比例：14.0%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;个人：8%</div>\n\n<div>非深户公司缴交养老保险比例：13.0%&nbsp;&nbsp;&nbsp;个人8%</div>\n\n<div>医疗保险公司缴交比例：6.2%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;个人2%</div>\n\n<div><!--StartFragment -->失业保险公司缴交比例：1.0%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;个人0.5%<br />\n工伤保险公司缴交比例：0.28%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;个人0<br />\n生育保险公司缴交比例：0.5%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;个人0</div>\n</div>\n","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.77308610237860009,"Rank":"2","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0668","FaqCode":"hris_CB-0668","FaqStaffType":"[正式]","FaqPlace":"[深圳]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d60707434dcf12a783b4dff8083e","FaqHot":"93","FaqTitle":"公司每月什么时候给员工进行缴纳社保、公积金？","FaqAnswer":"你好，公司一般是每月的中下旬给员工缴纳当月社保公积金，因为社保公积金管理中心有个接收时间，所以一般公积金每个月21号左右到帐，社保要到月底才会到帐。","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.66150167201099364,"Rank":"3","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0880","FaqCode":"hris_CB-0880","FaqStaffType":"[正式]","FaqPlace":"[深圳]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d85f2fe2a6c89cec5104be799def","FaqHot":"102","FaqTitle":"如何查询自己的社保信息呢？","FaqAnswer":"<p>你好，1、登录深圳市社会保险基金管理中心网址：http://www.szsi.gov.cn/index_16259.htm，可输入你的社保电脑号、身份证号进行社保帐户余额查询。&nbsp;2、可拨打深圳社保局热线电话：12333进行查询。&nbsp;3、可登录社会保险服务个人网页：https://e.szsi.gov.cn/siservice/，注册后即可查询缴费明细。（如注册时需要申报手机号可于每周五联系HR社保管理karliewang(王晓然)申报）</p>\n","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.53934411601877175,"Rank":"4","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0026","FaqCode":"hris_CB-0026","FaqStaffType":"[正式]","FaqPlace":"[深圳]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d5b09d2d1fb61ce4d504314b859b","FaqHot":"10","FaqTitle":"离职后是否可以以个人身份去缴交社保？","FaqAnswer":"你好，深圳地区只有深圳户口的离职员工是可以以个人身份去社保局进行缴交的。","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.70827896228943532,"Rank":"5","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000091","Faq2CategoryId":"hris_2_00000091","FaqId":"hris_S-0246","FaqCode":"hris_S-0246","FaqStaffType":"[正式]","FaqPlace":"[全国]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1576a9c3457620b00ab844d4dd0b3c66","FaqHot":"20","FaqTitle":"入职北京地区，如何填写社保公积金帐号？","FaqAnswer":"<p>你好，如果以前没有在北京缴纳过社保、公积金，填写0即可。&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />\n在北京缴纳过社保公积金的：社保帐号为身份证号，公积金帐号为身份证号后面加00&nbsp;例：“10112319999999123400”。</p>\n","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0}],"RtxAnswerInfo":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"RtxSessionInfo":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":"yapingwang","Id":0}
其中涉及到一个返回类型responsetype字段，用于标识返回的FAQ问答结果类型，其中包括以下几种类型：
1.	精确命中，一个ResAnswer
2.	推荐top 5，<= 5个ResAnswer，并根据ResAnswer中Score做好排序
3.	无法回答，三个相关的ResAnswer
8.	闲聊 （需要连接外网）

测试环境测试接口（暂时）：
post：http://10.12.91.105:5000/autoreply


相关文件介绍：
config.py : 包括了是否打开闲聊，最大句子长度，top_k大小等参数设定
request.py: 运行eval_sentence_rest.py后，测试rest服务是否正常
accuracy_test.py: 对比BM25算法测试模型在测试集上的表现
stacking.py: 根据w2v和BM25的scores值，利用逻辑回归将两个分数联合作为句子相似度的判别依据。
FAQ_with_sim.csv: 原始FAQ库
public_FAQ.csv: 公共的问答库

数据集的格式：
.csv或excel文件（需要将cal_simlation.py中的read_csv 改为 read_excel）

原理介绍：
1.首先使用闲聊库和业务问答库文本作为训练集，训练处二分类的判别问题是否业务问题的网络。
2.使用BM25以及平均词向量加上cosine距离衡量句子间相似度。
3.使用stacking融合模型得到最终的句子相似度得分。
4.优先搜索私有库，若找不到相关答案，则搜索公共库。

TODO:
1.使用L5在内网部署闲聊机器人接口。
2.优化代码结构，减少响应时间。
3.增加stacking的特征，进一步提高准确度。

