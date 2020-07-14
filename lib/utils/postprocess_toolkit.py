# Author: Lu Tongyu
# Please modify the items for yourself
# date: 20200707
import torch
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import numpy as np
import math
import random

import json
import pickle

from matplotlib.font_manager import *  
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            if int(token) in int2word:
                word = int2word[int(token)]
            else:
                word = '<UNK>'
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)
    
    return sentences


import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def computebleu(sentences, targets):
    score = 0 
    if len(sentences) < len(targets):
        #print(sentences)
        #print(targets)
        to_add = len(targets)-len(sentences)
        for i in range(to_add):
            sentences.append('<PAD>')
    else:
        assert (len(sentences) == len(targets))


    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp 

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))
    
    return score

def mask_sequence(ref_seq, head_seq, mask_index, head_len = 10, cut_down = False):
    global device
    ref_len = ref_seq.shape[1]
    ref_shape = ref_seq.shape
    masked_seq = (torch.LongTensor(torch.ones(ref_shape).long())*mask_index).to(device)
    for i in range(head_len):
        masked_seq[:,i] = head_seq[:,i]
    if cut_down:
        masked_seq = masked_seq[0:head_len]
    return masked_seq

def schedule_sampling(step,summary_steps):
    return 1-0.8*step/summary_steps

def plot_attention(labels, attention_matrices_list):
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_matrices_list.cpu().detach().numpy())
    plt.yticks(range(len(labels)), labels)
    plt.xticks(range(len(labels)), labels)
    plt.show()
