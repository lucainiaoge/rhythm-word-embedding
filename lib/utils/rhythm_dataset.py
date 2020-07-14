# Author: Lu Tongyu
# 20200707
# 


import torch
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms

import numpy as np
import math
import random

import json
import pickle


import sys
import os
sys.path.append("..")
sys.path.append("/content/drive/My Drive/Colab Notebooks/music_GAN_rhythm_seed")
sys.path.append("/content/drive/My Drive/Colab Notebooks/music_GAN_rhythm_seed/data_loader")
from data_loader.Nottingham_database_preprocessor import *
from data_loader.Nottingham_database_preprocessor_util import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelTransform(object):
    def __init__(self, size, pad):
        self.size = size
        self.pad = pad

    def __call__(self, label):
        label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
        return label

def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x

class RhythmDataset(data.Dataset):
    def __init__(self, config, set_name, pad=False):
        self.root = config.data_path
        self.max_output_len = config.max_output_len
        self.word2int, self.int2word = self.get_dictionary()
        self.max_id = max(self.int2word)
        print('max index in dict is ',self.max_id)

        self.dict_size = len(self.word2int)
        self.vocab_size = self.max_id+1
        self.data = []
        self.data_class = []
        self.data_grouping = []
        self.data_freq = []
        self.data_sample_weight = []
        self.data_meter_freq = []
        self.data_meter_weight = []
        with open(os.path.join(self.root, f'{set_name}'), "rb") as f:
            self.data=pickle.load(f)
            print (f'{set_name} dataset size: {len(self.data)}')
        
        self.word2freq, self.int2freq = self.get_word_freq()
        self.get_data_freq_and_weight()

        self.class_contents_int2class = config.class_contents_int2class
        self.class_contents_class2int = config.class_contents_class2int
        self.meter2freq = {}
        self.grouping_int2class = config.grouping_int2class
        self.grouping_class2int = config.grouping_class2int
        self.binary_list = config.binary_list
        self.ternary_list = config.ternary_list
        self.quintet_list = config.quintet_list

        self.word_idx2meter_idx = {}
        self.word_idx2grouping_idx = {}

        self.get_contents_and_grouping_dictionary()
        self.align_classes_for_data()
        
        self.pad = pad
        self.mask_prob = config.mask_prob
        self.generate_mode = False
        self.bias_tokens_n = config.bias_tokens_n
        self.head_len = config.head_len
        self.transform = LabelTransform(self.max_output_len, self.word2int['<PAD>'])
        self.transform_class = LabelTransform(self.max_output_len, 0)

    def get_dictionary(self):
        with open(os.path.join(self.root+'/rhythm_dict', f'vocab_word2int.json'), "r") as f:
            word2int = json.load(f)
        with open(os.path.join(self.root+'/rhythm_dict', f'vocab_int2word.json'), "r") as f:
            int2word = json.load(f, object_hook=jsonKeys2int)
        return word2int, int2word

    def get_word_freq(self):
        word2freq = {}
        int2freq = {}
        for index in self.int2word:
            int2freq.update({index:0})
            word2freq.update({self.int2word[index]:0})
        for sen in self.data:
            for word in sen:
                this_word_id = self.word2int[word]
                int2freq[this_word_id] += 1
                word2freq[word] += 1
        return word2freq, int2freq

    def get_data_freq_and_weight(self,var_factor = 0.5):
        self.data_freq = []
        self.data_sample_weight = []
        for sen in self.data:
            this_freq = 0
            this_len = len(sen)
            for word in sen:
                this_freq += self.word2freq[word]/this_len
            self.data_freq.append(this_freq)
        ave_freq = sum(self.data_freq)/len(self.data)
        for fre in self.data_freq:
            this_w = math.exp(-1*var_factor*(fre/ave_freq))
            self.data_sample_weight.append(this_w)
        sum_w = sum(self.data_sample_weight)
        for i,fre in enumerate(self.data_sample_weight):
            self.data_sample_weight[i] = fre/sum_w*len(self.data)

    def class_str2grouping_idx(self,class_str):
        if class_str == 'controller':
            return 0
        elif class_str in self.binary_list:
            return self.grouping_class2int['binary']
        elif class_str in self.ternary_list:
            return self.grouping_class2int['ternary']
        elif class_str in self.quintet_list:
            return self.grouping_class2int['quintet']
        else:
            return self.grouping_class2int['others']
    
    def class_idx2grouping_idx(self,class_idx):
        class_str = self.class_contents_int2class[class_idx]
        return self.class_str2grouping_idx(class_str)

    def get_contents_and_grouping_dictionary(self):
        last_index = len(self.class_contents_int2class)-1
        for index in range(self.vocab_size):
            if index not in self.int2word:
                self.word_idx2meter_idx.update({index:last_index})             
                continue
            else:
                find_flag = 0
                for index_meter in self.class_contents_int2class:
                    this_class = self.class_contents_int2class[index_meter]
                    word = self.int2word[index]
                    if this_class in word:
                        self.word_idx2meter_idx.update({index:index_meter})
                        find_flag = 1
                        break
                if not find_flag:
                    self.word_idx2meter_idx.update({index:0})
            
            this_grouping = self.class_idx2grouping_idx(self.word_idx2meter_idx[index])
            self.word_idx2grouping_idx.update({index:this_grouping})

    def align_classes_for_data(self,var_factor=2):
        self.data_meter = []
        self.data_grouping = []
        self.data_meter_freq = []
        for sen in self.data:
            this_sen_meter = []
            this_sen_grouping = []
            for word in sen:
                this_word_id = self.word2int[word]
                this_word_meter = self.word_idx2meter_idx[this_word_id]
                this_word_grouping = self.word_idx2grouping_idx[this_word_id]
                this_sen_meter.append(this_word_meter)
                this_sen_grouping.append(this_word_grouping)
            self.data_meter.append(this_sen_meter)
            self.data_grouping.append(this_sen_grouping)
            if this_sen_meter[1] not in self.meter2freq:
                self.meter2freq.update({this_sen_meter[1]:1})
            else:
                self.meter2freq[this_sen_meter[1]] += 1
        for id,met in enumerate(self.data_meter):
            this_meter_id = met[1]
            self.data_meter_freq.append(self.meter2freq[this_meter_id])
        ave_freq = sum(self.data_meter_freq)/len(self.data)
        for fre in self.data_meter_freq:
            this_w = math.exp(-1*var_factor*(fre/ave_freq))
            self.data_meter_weight.append(this_w)
        sum_w = sum(self.data_sample_weight)
        for i,fre in enumerate(self.data_meter_weight):
            self.data_meter_weight[i] = fre/sum_w*len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        global device
        sentence = self.data[Index]
        sentence_meter = self.data_meter[Index]
        sentence_grouping = self.data_grouping[Index]
        sentence_idx = []
        target = []
        masked_pos = []
        word_freq = []
        for index,word in enumerate(sentence):
            word_sen = word
            if self.mask_prob>0 and index>self.head_len:
                if random.random()<self.mask_prob:
                    temp_rand = random.random()
                    if temp_rand<1:
                        word_sen = '<MASK>'
                    elif temp_rand>=1:
                        word_sen_id = int((self.max_id-self.bias_tokens_n)*random.random())+self.bias_tokens_n
                        word_sen = self.int2word[word_sen_id]
            if self.generate_mode and index>self.head_len:
                word_sen = '<MASK>'
            if (word_sen in self.word2int.keys()):
                if word_sen=='<MASK>':
                    masked_pos.append(1)
                else:
                    masked_pos.append(0)
                sentence_idx.append(self.word2int[word_sen])
            else:
                masked_pos.append(0)
                sentence_idx.append(self.word2int['<UNK>'])
            if (word in self.word2int.keys()):
                target.append(self.word2int[word])
            else:
                target.append(self.word2int['<UNK>'])

        if self.pad:
            sentence_idx = np.asarray(sentence_idx)
            sentence_idx = self.transform(sentence_idx)
            target = np.asarray(target)
            target = self.transform(target)
            sentence_meter = np.asarray(sentence_meter)
            sentence_meter = self.transform_class(sentence_meter)
            sentence_grouping = np.asarray(sentence_grouping)
            sentence_grouping = self.transform_class(sentence_grouping)    

        for sen_id in target:
            word_freq.append(self.int2freq[sen_id])

        sentence_idx = torch.LongTensor(sentence_idx).to(device)
        target = torch.LongTensor(target).to(device)
        sentence_meter = torch.LongTensor(sentence_meter).to(device)
        sentence_grouping = torch.LongTensor(sentence_grouping).to(device)
        masked_pos = torch.LongTensor(masked_pos).to(device)
        word_freq = torch.LongTensor(word_freq).to(device)
        sen_freq = self.data_freq[Index]
        sen_weight = self.data_sample_weight[Index]
        meter_freq = self.data_meter_freq[Index]
        meter_weight = self.data_meter_weight[Index]
        output_package = {'id':Index,'sentence_idx':sentence_idx,'target':target,'sentence_meter':sentence_meter,
                          'sentence_grouping':sentence_grouping,'masked_pos':masked_pos,'word_freq':word_freq,
                          'sen_freq':sen_freq,'sen_weight':sen_weight,'meter_freq':meter_freq,'meter_weight':meter_weight}

        return output_package


