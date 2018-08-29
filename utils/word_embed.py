# -*- coding: utf-8 -*-
"""
generate a list of word vectors for a sequence of words
"""

import numpy as np

##wait to modify:  all zero down error
def embed(words_list, word2vec, max_doc_len):
    vec_list = []
    miss_word_num = 0
    for words in words_list:
        word_vec = []
        for i in range(max_doc_len):
            if i < len(words):
                word = words[i]
                if word in word2vec.vocab:
                    embed = True
                    word_vec.append(word2vec.word_vec(word).tolist())
                else:
                    miss_word_num += 1
                    #找不到词向量，均匀随机分布
                    # word_vec.append(np.random.rand(word2vec.vector_size).tolist())
                    word_vec.append(np.zeros(word2vec.vector_size).tolist())
            else:
                #补0
                word_vec.append(np.zeros(word2vec.vector_size).tolist())
        vec_list.append(word_vec)
    return vec_list
def embed_sentence(sentence, word2vec, max_sentence_len):
    word_vec = []
    miss_word_num = 0
    for i in range(max_sentence_len):
        if i < len(sentence):
            word = sentence[i]
            if word in word2vec.vocab:
                word_vec.append(word2vec.word_vec(word).tolist())
            else:
                miss_word_num += 1
                word_vec.append(np.zeros(word2vec.vector_size).tolist())
                # word_vec.append(list(np.random.rand(word2vec.vector_size)))
        else:
            word_vec.append(np.zeros(word2vec.vector_size).tolist())
    return word_vec,miss_word_num
