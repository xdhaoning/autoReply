#coding = utf-8
def reply2json(reply,responsetype):
    AnswerList = []
    if responsetype is 3 or responsetype is 8:
        cnt = 1
    else:
        cnt = len(reply[0])
    for i in range(cnt):
        if responsetype is 3 or responsetype is 8:
            standard_question = None
            standard_ans = reply[0]
            score = None
            module = " "
        else:
            standard_question = reply[0][i]
            standard_ans = reply[1][i]
            score = reply[2][i]
            module = str(reply[3][i])
        rank = i + 1
        faq_dict = {
        "RtxAnswerInfo_ResId":0,"Score":score,"Rank":rank,"Method":"AVG","FaqItemList":None,"FaqItem":{
        "RtxAnswerInfo_Res_AnsId":0,
        "Faq1CategoryId":"hris_2_00000141",
        "Faq2CategoryId":"hris_2_00000141",
        "FaqId":"hris_" + module,
        "FaqCode":"hris_" + module,
        "FaqStaffType":"[正式]",
        "FaqPlace":"[深圳]",
        "FaqUrl":"http://s3.oa.com/kms/multidoc/kms_multidoc_knowledge/kmsMultidocKnowledge.do?method=view&fdId=1539d5fd83b0532f0611b6d4172854aa",
        "FaqHot":"49",
        "FaqTitle":standard_question,
        "FaqAnswer":standard_ans,
        "RtxAnswerInfo_Res_Ans":None,
        "AddTime":"2017-12-07T14:55:57.75+08:00",
        "AddMan":None,
        "Id":0}}
        AnswerList.append(faq_dict)
    ListResAnswer = {"RtxAnswerInfoId":0,"Cnt":cnt,
    "HuaShu":"为您找到以下相关问题，点击可获取对应答案：",
    "Flag_HuaShu":"1",
    "MoreHuaShu":None,
    "MoreTag":None,
    "IsSeficiencyFaq":None,
    "Area":None,
    "AnswerList": AnswerList}
    final_reply = {
    "RtxSessionInfoId":3055,
    "Code":0,
    "Msg":"OK",
    "BusinessId":"HRIS",
    "SessionId":"HRIS_yapingwang_20171207145557593",
    "ResponseType":responsetype,
    "ListResAnswerList":None,
    "ListResAnswer":ListResAnswer}
    return final_reply