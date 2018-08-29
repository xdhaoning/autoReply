# -*- coding:utf8 -*-
import math
import numpy as np
def average_vector(sentence_vector,sentence_len):
    v = np.array(sentence_vector)
    average = sum(v)/sentence_len
    return average

def weighting_average_vector(sentence_vector,tfidf_bow,sentence_len):
    v = np.array(sentence_vector)
    sum_bow = 0
    average_vec = 0
    for bow in tfidf_bow:
        sum_bow = sum_bow + bow[1]
    for i in range(len(tfidf_bow)):
        word_vec = v[i]
        idf_score = tfidf_bow[i][1] / sum_bow
        average_vec = average_vec + word_vec * idf_score
    return average_vec


def cosine_distance(vector1,vector2,embedding_size):
    vector1 = np.reshape(vector1,newshape=[1,embedding_size])
    vector2 = np.reshape(vector2, newshape=[embedding_size,1])
    score = (np.dot(vector1,vector2)/(np.linalg.norm(vector1) * np.linalg.norm(vector2)))[0][0]
    if score <= 1:
        return score
    else:
        return 0
    # return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def euclidean_distance(vector1, vector2,embedding_size):
    # sum = 0.0
    vector1 = np.reshape(vector1,newshape=[1,embedding_size])
    vector2 = np.reshape(vector2, newshape=[1,embedding_size])
    return np.linalg.norm(vector1 - vector2)
    # if len(vector1) == len(vector2):
    #     for i in range(len(vector1)):
    #         delta = vector1[i] - vector2[i]
    #         sum += delta * delta
    #     return math.sqrt(sum)
    # else:
    #     raise (Exception('Vector dimmension not equal'))


def distance(vector1,vector2,embedding_size,method = 'cosine'):
    if method == 'cosine':
        return cosine_distance(vector1,vector2,embedding_size)
    elif method == 'euclidean':
        return euclidean_distance(vector1,vector2,embedding_size)
    else:
        #其他计算距离方式
        pass