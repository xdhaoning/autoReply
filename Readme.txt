�����⣺
	1.anaconda3
	2.tensorflow (����ʹ��ҵ���ҵ���������ֹ��ܣ�����Ҫ)
	3.gensim
	4.jieba (����ʹ��AiLab�ķִ�)


1.������ʹ�õ�Ϊ����wiki�ٿ�Ԥѵ���õ�ģ�ͣ���ַΪhttp://pan.baidu.com/s/1dFeNNK9
2.�ڷ����������� eval_sentence_rest.py�ļ��󼴿ɿ���5000�˿ڣ����ع۲�Ч�������� eval_sentence(���滯).py�� ��ز������ö�����config.py�
3.��ҵ������BiLSTM����ģ�Ͳο��� http://git.code.oa.com/AI_algorithm/QuickReply

����Rest��������ݸ�ʽ��
ʾ�����£�5000�˿ڽ��յ�json��ʽ������󣬸���json��ʽ�ķ���
request������
{"businessid":"HRIS","sessionid":"HRIS_yapingwang_20171207145557593","userprofile":{"rtx_ename":"yapingwang","rtx_name":"����ƽ","worksitecode":"����","emptypecode":"��ʽ","managelevel":"","majorlevel":"","issecretary":"N"},"userquestion":{"content":"�籣"}}

response�����أ�
{"RtxSessionInfoId":3055,"Code":0,"Msg":"OK","BusinessId":"HRIS","SessionId":"HRIS_yapingwang_20171207145557593","ResponseType":2,"ListResAnswerList":null,"ListResAnswer":{"RtxAnswerInfoId":0,"Cnt":5,"HuaShu":"Ϊ���ҵ�����������⣬����ɻ�ȡ��Ӧ�𰸣�","Flag_HuaShu":"1","MoreHuaShu":null,"MoreTag":null,"IsSeficiencyFaq":null,"Area":null,"AnswerList":[{"RtxAnswerInfo_ResId":0,"Score":0.64130109004241775,"Rank":"1","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0603","FaqCode":"hris_CB-0603","FaqStaffType":"[��ʽ]","FaqPlace":"[����]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d5fd83b0532f0611b6d4172854aa","FaqHot":"49","FaqTitle":"��˾�͸��˵��籣�ɽ��������Ƕ��٣�","FaqAnswer":"<p>��ã���ο���</p>\n\n<div>\n<div>���˾�ɽ����ϱ��ձ�����14.0%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;���ˣ�8%</div>\n\n<div>�����˾�ɽ����ϱ��ձ�����13.0%&nbsp;&nbsp;&nbsp;����8%</div>\n\n<div>ҽ�Ʊ��չ�˾�ɽ�������6.2%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;����2%</div>\n\n<div><!--StartFragment -->ʧҵ���չ�˾�ɽ�������1.0%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;����0.5%<br />\n���˱��չ�˾�ɽ�������0.28%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;����0<br />\n�������չ�˾�ɽ�������0.5%&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;����0</div>\n</div>\n","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.77308610237860009,"Rank":"2","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0668","FaqCode":"hris_CB-0668","FaqStaffType":"[��ʽ]","FaqPlace":"[����]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d60707434dcf12a783b4dff8083e","FaqHot":"93","FaqTitle":"��˾ÿ��ʲôʱ���Ա�����н����籣��������","FaqAnswer":"��ã���˾һ����ÿ�µ�����Ѯ��Ա�����ɵ����籣��������Ϊ�籣��������������и�����ʱ�䣬����һ�㹫����ÿ����21�����ҵ��ʣ��籣Ҫ���µײŻᵽ�ʡ�","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.66150167201099364,"Rank":"3","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0880","FaqCode":"hris_CB-0880","FaqStaffType":"[��ʽ]","FaqPlace":"[����]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d85f2fe2a6c89cec5104be799def","FaqHot":"102","FaqTitle":"��β�ѯ�Լ����籣��Ϣ�أ�","FaqAnswer":"<p>��ã�1����¼��������ᱣ�ջ������������ַ��http://www.szsi.gov.cn/index_16259.htm������������籣���Ժš����֤�Ž����籣�ʻ�����ѯ��&nbsp;2���ɲ��������籣�����ߵ绰��12333���в�ѯ��&nbsp;3���ɵ�¼��ᱣ�շ��������ҳ��https://e.szsi.gov.cn/siservice/��ע��󼴿ɲ�ѯ�ɷ���ϸ������ע��ʱ��Ҫ�걨�ֻ��ſ���ÿ������ϵHR�籣����karliewang(����Ȼ)�걨��</p>\n","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.53934411601877175,"Rank":"4","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000141","Faq2CategoryId":"hris_2_00000141","FaqId":"hris_CB-0026","FaqCode":"hris_CB-0026","FaqStaffType":"[��ʽ]","FaqPlace":"[����]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d5b09d2d1fb61ce4d504314b859b","FaqHot":"10","FaqTitle":"��ְ���Ƿ�����Ը������ȥ�ɽ��籣��","FaqAnswer":"��ã����ڵ���ֻ�����ڻ��ڵ���ְԱ���ǿ����Ը������ȥ�籣�ֽ��нɽ��ġ�","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},{"RtxAnswerInfo_ResId":0,"Score":0.70827896228943532,"Rank":"5","Method":"AVG","FaqItemList":null,"FaqItem":{"RtxAnswerInfo_Res_AnsId":0,"Faq1CategoryId":"hris_2_00000091","Faq2CategoryId":"hris_2_00000091","FaqId":"hris_S-0246","FaqCode":"hris_S-0246","FaqStaffType":"[��ʽ]","FaqPlace":"[ȫ��]","FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1576a9c3457620b00ab844d4dd0b3c66","FaqHot":"20","FaqTitle":"��ְ���������������д�籣�������ʺţ�","FaqAnswer":"<p>��ã������ǰû���ڱ������ɹ��籣����������д0���ɡ�&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<br />\n�ڱ������ɹ��籣������ģ��籣�ʺ�Ϊ���֤�ţ��������ʺ�Ϊ���֤�ź����00&nbsp;������10112319999999123400����</p>\n","RtxAnswerInfo_Res_Ans":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"ResultAnswer":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0}],"RtxAnswerInfo":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":null,"Id":0},"RtxSessionInfo":null,"AddTime":"2017-12-07T14:55:57.75+08:00","AddMan":"yapingwang","Id":0}
�����漰��һ����������responsetype�ֶΣ����ڱ�ʶ���ص�FAQ�ʴ������ͣ����а������¼������ͣ�
1.	��ȷ���У�һ��ResAnswer
2.	�Ƽ�top 5��<= 5��ResAnswer��������ResAnswer��Score��������
3.	�޷��ش�������ص�ResAnswer
8.	���� ����Ҫ����������

���Ի������Խӿڣ���ʱ����
post��http://10.12.91.105:5000/autoreply


����ļ����ܣ�
config.py : �������Ƿ�����ģ������ӳ��ȣ�top_k��С�Ȳ����趨
request.py: ����eval_sentence_rest.py�󣬲���rest�����Ƿ�����
accuracy_test.py: �Ա�BM25�㷨����ģ���ڲ��Լ��ϵı���
stacking.py: ����w2v��BM25��scoresֵ�������߼��ع齫��������������Ϊ�������ƶȵ��б����ݡ�
FAQ_with_sim.csv: ԭʼFAQ��
public_FAQ.csv: �������ʴ��

���ݼ��ĸ�ʽ��
.csv��excel�ļ�����Ҫ��cal_simlation.py�е�read_csv ��Ϊ read_excel��

ԭ����ܣ�
1.����ʹ�����Ŀ��ҵ���ʴ���ı���Ϊѵ������ѵ������������б������Ƿ�ҵ����������硣
2.ʹ��BM25�Լ�ƽ������������cosine����������Ӽ����ƶȡ�
3.ʹ��stacking�ں�ģ�͵õ����յľ������ƶȵ÷֡�
4.��������˽�п⣬���Ҳ�����ش𰸣������������⡣

TODO:
1.ʹ��L5�������������Ļ����˽ӿڡ�
2.�Ż�����ṹ��������Ӧʱ�䡣
3.����stacking����������һ�����׼ȷ�ȡ�

