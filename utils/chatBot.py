import urllib.request
import hashlib
import urllib
import json
import time

url_preffix = 'https://api.ai.qq.com/fcgi-bin/'


def setParams(array, key, value):
    array[key] = value


def genSignString(parser):
    uri_str = ''
    for key in sorted(parser.keys()):
        if key == 'app_key':
            continue
        uri_str += "%s=%s&" % (key, urllib.parse.quote(str(parser[key]), safe=''))
    sign_str = uri_str + 'app_key=' + parser['app_key']
    hash_md5 = hashlib.md5(sign_str.encode("utf-8"))
    return hash_md5.hexdigest().upper()


class AiPlat(object):
    def __init__(self, app_id, app_key):
        self.app_id = app_id
        self.app_key = app_key
        self.data = {}

    def invoke(self, params):
        self.url_data = urllib.parse.urlencode(params).encode('utf-8')
        req = urllib.request.Request(self.url, self.url_data)
        # try:
        rsp = urllib.request.urlopen(req)
        str_rsp = rsp.read()
        dict_rsp = json.loads(str_rsp)
        return dict_rsp

    def getNlpTextTrans(self, text):
        self.url = url_preffix + 'nlp/nlp_textchat'
        setParams(self.data, 'app_id', self.app_id)
        setParams(self.data, 'app_key', self.app_key)
        setParams(self.data, 'time_stamp', int(time.time()))
        setParams(self.data, 'nonce_str', int(time.time()))
        setParams(self.data, 'session', '10000')
        setParams(self.data, 'question', text)
        sign_str = genSignString(self.data)
        setParams(self.data, 'sign', sign_str)
        return self.invoke(self.data)