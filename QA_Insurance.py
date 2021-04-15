
# coding: utf-8
# @time     :2020/09/04
# @author   :Shaoran
# @function :QA baseline on insuranceqa
# In[1]:


import sys
import os
import json
import unicodedata
import jieba
import numpy as np
from gensim.models import KeyedVectors
# from gensim.models import Word2Vec
import queue as Q

from transformers import BertConfig, BertTokenizer, TFBertModel

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
if len(physical_devices) != 0:
    for pd in physical_devices:        
        tf.config.experimental.set_memory_growth(pd, True)
        


# In[2]:


def stopwordslist(filepath='./hit_stopwords.txt'):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


# 对句子进行分词并去停用词
def seg_sentence(sentence, stopwords, use_stopwords=False):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = []
    for word in sentence_seged:
        if use_stopwords:
            if word not in stopwords:
                if word != '\t':
                    outstr.append(word)
        else:
            if word != '\t':
                outstr.append(word)            
    return outstr

stopwords = stopwordslist()
# print(stopwords)
print(seg_sentence('法律要求残疾保险吗？', stopwords, use_stopwords=False))


# In[3]:


def read_corpus(trainQ_corpus_path='./data/insuranceqa-corpus-zh-release/corpus/pool/train.json',
               answers_path ='./data/insuranceqa-corpus-zh-release/corpus/pool/answers.json'):
    with open(trainQ_corpus_path, mode='r') as fp:
        trainQ_corpus_json = json.load(fp)


    with open(answers_path, mode='r') as fc:
        answers_json = json.load(fc)

    q_list = []
    a_list = []
    for q in trainQ_corpus_json.keys():
#         q_list.append(trainQ_corpus_json[q]['zh'])
        q_list.append(''.join(seg_sentence(trainQ_corpus_json[q]['zh'], stopwords, use_stopwords=False)))
        tmp_a = []
        for a_ in trainQ_corpus_json[q]['answers']:
            tmp_a.append(unicodedata.normalize('NFKC', answers_json[a_]['zh']).strip())
        a_list.append(tmp_a)

    return q_list, a_list

qlist,alist = read_corpus()
print(len(qlist), len(alist))
print("Q:")
print(qlist[:10])
print('A:')
print(alist[:5])


print(len(qlist))
print(len(alist))

bert_model = TFBertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

if not os.path.exists('./Q_bert_vecters_chn.np'):
    print('Using bert to cal vecs....')
    X_bert = np.zeros((len(qlist), 768))
    for i, q in enumerate(qlist):
        input_id = tokenizer.encode(q, return_tensors='tf')
        
        sentences_embedding = tf.squeeze(tf.math.reduce_sum(
            bert_model(input_id).last_hidden_state[:,1:, :], axis=1), axis=0)
        X_bert[i] = sentences_embedding.numpy()
        

    X_bert.tofile('./Q_bert_vecters_chn.np')
    
else:
    print('reloading bert vecs...')
    X_bert = np.fromfile('./Q_bert_vecters_chn.np')
    X_bert = X_bert.reshape((-1, 768))
    print(X_bert.shape)
    assert X_bert.shape[1] == 768, 'check the shape with X_bert'
    print(X_bert[:2])


def cosineSimilarity(vec1,vec2):
    return np.dot(vec1,vec2.T)/(np.sqrt(np.sum(vec1**2))*np.sqrt(np.sum(vec2**2)))

word_doc = dict()
for i,q in enumerate(qlist):
    words = seg_sentence(q, stopwords, use_stopwords=True)
    for w in set(words):
        if w not in word_doc:
            word_doc[w] = set([])
        word_doc[w] = word_doc[w] | set([i])
inverted_idx = word_doc 


# In[8]:


print(len(inverted_idx))
vocab_count = inverted_idx.keys()
print(len(vocab_count))
    
word2vec = KeyedVectors.load_word2vec_format('./zhwiki.word2vec/zhwiki.word2vec.bin', binary=True)


def make_related_words(vocab_count, embedding, topk=5, file_name='./related_words_chn.json'):
    related_word_dict = {}
    for vc in vocab_count:
        if vc not in embedding:
            continue
        
        queq = Q.PriorityQueue()
        for vcc in vocab_count:
            if vcc not in embedding:
                continue
            cs = cosineSimilarity(embedding[vc], embedding[vcc])
            queq.put((-1 * cs,vcc))
        i = 0
        top_words = [] 
        while(i < topk and not queq.empty()):
            top_words.append(queq.get()[1])
            i += 1
            
        related_word_dict[vc] = ' '.join(top_words)
        
    with open(file_name, mode='w', encoding='UTF-8') as fw:
        fw.write(json.dumps(related_word_dict, ensure_ascii=False))
        
    return related_word_dict

if not os.path.exists('./related_words_chn_ITS.json'): 
    print('constructing related words...')
    related_word_dict = make_related_words(vocab_count, word2vec, file_name='./related_words_chn_ITS.json')
else:
    with open('./related_words_chn_ITS.json', mode='r', encoding='UTF-8') as fr:
        print('reloading related words...')
        ff = fr.readlines()
        related_word_dict = json.loads(ff[0])
        
print(related_word_dict)


def get_related_words(related_word_dict):

    related_words = {}
    for word in related_word_dict:
        re_words = related_word_dict[word]
        re_words = re_words.strip().split(' ')
        related_words[word] = re_words[1:] 
    return related_words

related_words = get_related_words(related_word_dict)

def getCandidate(query, inverted_idx, related_words):
    searched = set()
    for w in query:
        if w not in inverted_idx:
            continue
        if len(searched) == 0:
            searched = set(inverted_idx[w])
        else:
            searched = searched | set(inverted_idx[w])
        if w in related_words:
#             print('w:',w)
            for similar in related_words[w]:
#                 print(similar)
                if similar in inverted_idx:
                    searched = searched | set(inverted_idx[similar])
    return searched


def calcu_Bert_vec(sentence):
    input_id = tokenizer.encode(sentence.lower(), return_tensors='tf')
    
    sentences_embeddings = bert_model(input_id,output_hidden_states=True).last_hidden_state
    emb = np.squeeze(sentences_embeddings.numpy(), axis=0)[-2]

#     print(bert_model(input_id).last_hidden_state.numpy().shape)
    
    return emb

def calcu_similarity(sentence_1, sentence_2):
    emb_1 = calcu_Bert_vec(sentence_1)
    emb_2 = calcu_Bert_vec(sentence_2)
    
    return cosineSimilarity(emb_1,emb_2)

# print(calcu_similarity('给定用户输入的问题 query, 返回最有可能的TOP 5问题。这里面需要做到以下几点：', '两个高原构成其余30%的表面地区，一个在行星的北半球，另一个正好在赤道的南边。'))


# In[15]:

def get_top_results_bert(query, X_bert, stopwords, topk):
    
    query_ = seg_sentence(query, stopwords, use_stopwords=True)
    
    top = topk
    
    input_id = tokenizer.encode(query, return_tensors='tf')

    sentences_embedding = tf.squeeze(tf.math.reduce_sum(
        bert_model(input_id).last_hidden_state, axis=1), axis=0)
    query_emb = sentences_embedding.numpy()
    
    
     
    results = Q.PriorityQueue()
    searched = getCandidate(query_, inverted_idx, related_words)
#     print(searched)
    for candidate in searched:
        result = cosineSimilarity(query_emb, X_bert[candidate])
        #print(result)
        results.put((-1 * result,candidate))
    top_idxs = [] 
    top_similarity = []
    i = 0
    while i < top and not results.empty():
        top_idxs.append(results.get()[1])
        top_similarity.append(results.get()[0])
        i += 1
    
    return np.array(alist)[top_idxs], top_similarity, top_idxs 

query = '购买车辆保险' 

res_top, res_similarity, res_top_index = get_top_results_bert(query, X_bert, stopwords, topk=5)

if len(res_top) > 1:
    res_top1 = res_top[0]

else:
    res_top1 = res_top

if len(res_top1)>1:
    res_top1_random = np.random.choice(np.array(res_top1))

else:
    res_top1_random = res_top1

print(res_top1_random)

print(res_similarity)
print(np.array(qlist)[res_top_index])


'''
res_top, res_similarity, res_top_index = get_top_results_bert(query, X_bert, stopwords, inverted_idx, related_words, topk=10)
res_top1 = res_top[0]
res_top1_random = np.random.choice(np.array(res_top1))
print(res_top1_random)
print(res_similarity)
print(np.array(qlist)[res_top_index])
'''

