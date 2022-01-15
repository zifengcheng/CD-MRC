#-*-coding:GBK -*-
import sys
sys.path.append('..')
from os.path import join
import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from config import *
from utils.utils import *
import random


torch.manual_seed(TORCH_SEED)
torch.cuda.manual_seed_all(TORCH_SEED)
torch.backends.cudnn.deterministic = True


def build_train_data(configs, fold_id, shuffle=True):
    train_dataset = MyDataset(configs, fold_id, data_type='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=shuffle, collate_fn=bert_batch_preprocessing)
    return train_loader


def build_inference_data(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1000,
                                              shuffle=False, collate_fn=bert_batch_preprocessing)
    return data_loader


def get_ecpair(configs, fold_id, data_type):
    dataset = MyDataset(configs, fold_id, data_type)
    return dataset.pair



class MyDataset(Dataset):
    def __init__(self, configs, fold_id, data_type, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.split = configs.split

        self.data_type = data_type
        self.train_file = join(data_dir, self.split, TRAIN_FILE % fold_id)
        self.valid_file = join(data_dir, self.split, VALID_FILE % fold_id)
        self.test_file = join(data_dir, self.split, TEST_FILE % fold_id)

        self.batch_size = configs.batch_size
        self.epochs = configs.epochs

        self.bert_tokenizer = BertTokenizer.from_pretrained(configs.bert_cache_path)

        self.doc_couples_list, self.y_emotions_list, self.y_causes_list, \
        self.doc_len_list, self.doc_id_list, \
        self.bert_token_idx_list, self.bert_clause_idx_list, self.bert_segments_idx_list, \
        self.bert_token_lens_list,self.y_causes_list_1 = self.read_data_file(self.data_type)
        self.pair = self.get_pair1()

    def __len__(self):
        return len(self.y_emotions_list)

    def __getitem__(self, idx):
        doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
        doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
        bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
        bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
        y_causes_list_1 = self.y_causes_list_1[idx]
        if len(bert_token_idx[bert_token_idx.index(102)+1:]) > 400:
            bert_token_idx, bert_clause_idx, \
            bert_segments_idx, bert_token_lens, \
            doc_couples, y_emotions, y_causes, doc_len,y_causes_list_1 = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                          bert_segments_idx, bert_token_lens,
                                                                          doc_couples, y_emotions, y_causes, doc_len,y_causes_list_1)

        bert_token_idx = torch.LongTensor(bert_token_idx)
        bert_segments_idx = torch.LongTensor(bert_segments_idx)
        bert_clause_idx = torch.LongTensor(bert_clause_idx)

        assert doc_len == len(y_emotions)
        return doc_couples, y_emotions, y_causes, doc_len, doc_id, \
               bert_token_idx, bert_segments_idx, bert_clause_idx, bert_token_lens,y_causes_list_1

    def read_data_file(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file

        doc_id_list = []
        doc_len_list = []
        doc_couples_list = []
        y_emotions_list, y_causes_list = [], []
        y_causes_list_1 = []
        bert_token_idx_list = []
        bert_clause_idx_list = []
        bert_segments_idx_list = []
        bert_token_lens_list = []

        data_list = read_json(data_file)
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_emotions, doc_causes = zip(*doc_couples)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples = list(map(lambda x: list(x), doc_couples))
            doc_couples_list.append(doc_couples)

            y_emotions, y_causes = [], []
            zeros = []
            doc_clauses = doc['clauses']
            #print(doc_clauses)
            document = ''
            for i in range(doc_len):
                emotion_label = int(i + 1 in doc_emotions)
                cause_label = int(i + 1 in doc_causes)
                y_emotions.append(emotion_label)
                y_causes.append(cause_label)
                zeros.append(0)

                clause = doc_clauses[i]
                clause_id = clause['clause_id']
                assert int(clause_id) == i + 1
                document += '[CLS] ' + clause['clause'] + ' [SEP] '
                

            #indexed_tokens = self.bert_tokenizer.encode('[CLS]' + '找到文档中的情感子句比如高兴，悲伤，气愤，厌恶，恐惧，惊讶'+ '[SEP] '+ document.strip(), add_special_tokens=False)
            indexed_tokens = self.bert_tokenizer.encode('[CLS]' + '那个是情感子句'+ '[SEP] '+ document.strip(), add_special_tokens=False)

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            clause_indices.remove(0)
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            #assert len(clause_indices) == doc_len
            assert len(y_emotions) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            y_emotions_list.append(y_emotions)
            y_causes_list_1.append(y_causes)
            y_causes_list.append(0)  # find emotion
            negative = []
            
            for i,j in enumerate(y_emotions):
                if j==1:
                    c = zeros.copy()
                    question = '[CLS]' + '那个是情感子句' + doc_clauses[i]['clause'] + '对应的原因子句'+ '[SEP] '
                    question += document
                    for k,z in enumerate(doc_emotions):
                        if z== (i+1):
                            c[doc_causes[k]-1] = 1
                        #print(c)                 
                    indexed_tokens = self.bert_tokenizer.encode(question.strip(), add_special_tokens=False)

                    clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                    clause_indices.remove(0)
                    doc_token_len = len(indexed_tokens)

                    segments_ids = []
                    segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                    segments_indices.append(len(indexed_tokens))
                    for i in range(len(segments_indices)-1):
                        semgent_len = segments_indices[i+1] - segments_indices[i]
                        if i % 2 == 0:
                            segments_ids.extend([0] * semgent_len)
                        else:
                            segments_ids.extend([1] * semgent_len)

                    #assert len(clause_indices) == doc_len
                    assert len(c) == doc_len
                    assert len(segments_ids) == len(indexed_tokens)
                    bert_token_idx_list.append(indexed_tokens)
                    bert_clause_idx_list.append(clause_indices)
                    bert_segments_idx_list.append(segments_ids)
                    bert_token_lens_list.append(doc_token_len)
                    
                    doc_id_list.append(doc_id)
                    doc_len_list.append(doc_len)
                    doc_couples_list.append(doc_couples)
                    
                    #print('second is',c,doc_id)
                    y_causes_list_1.append(y_causes)
                    y_emotions_list.append(c)
                    y_causes_list.append(1)  # find cause
                else:
                    negative.append(i)
                                      
            # negative sampling
            if torch.rand(1) < 0.1:
                negative = random.sample(negative, 1)
                c = zeros.copy()
                question = '[CLS]' + '那个是情感子句' + doc_clauses[negative[0]]['clause'] + '对应的原因子句'+ '[SEP] '
                question += document

                indexed_tokens = self.bert_tokenizer.encode(question.strip(), add_special_tokens=False)

                clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                clause_indices.remove(0)
                doc_token_len = len(indexed_tokens)

                segments_ids = []
                segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                segments_indices.append(len(indexed_tokens))
                for i in range(len(segments_indices)-1):
                    semgent_len = segments_indices[i+1] - segments_indices[i]
                    if i % 2 == 0:
                        segments_ids.extend([0] * semgent_len)
                    else:
                        segments_ids.extend([1] * semgent_len)

                #assert len(clause_indices) == doc_len
                assert len(c) == doc_len
                assert len(segments_ids) == len(indexed_tokens)
                bert_token_idx_list.append(indexed_tokens)
                bert_clause_idx_list.append(clause_indices)
                bert_segments_idx_list.append(segments_ids)
                bert_token_lens_list.append(doc_token_len)

                doc_id_list.append(doc_id)
                doc_len_list.append(doc_len)
                doc_couples_list.append(doc_couples)

                #print('second is',c,doc_id)
                y_causes_list_1.append(y_causes)
                y_emotions_list.append(c)
                y_causes_list.append(1)  # find cause

            # cause query
            indexed_tokens = self.bert_tokenizer.encode('[CLS]' + '那个是原因子句'+ '[SEP] '+ document.strip(), add_special_tokens=False)

            clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            clause_indices.remove(0)
            doc_token_len = len(indexed_tokens)

            segments_ids = []
            segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
            segments_indices.append(len(indexed_tokens))
            for i in range(len(segments_indices)-1):
                semgent_len = segments_indices[i+1] - segments_indices[i]
                if i % 2 == 0:
                    segments_ids.extend([0] * semgent_len)
                else:
                    segments_ids.extend([1] * semgent_len)

            #assert len(clause_indices) == doc_len
            assert len(y_emotions) == doc_len
            assert len(segments_ids) == len(indexed_tokens)
            bert_token_idx_list.append(indexed_tokens)
            bert_clause_idx_list.append(clause_indices)
            bert_segments_idx_list.append(segments_ids)
            bert_token_lens_list.append(doc_token_len)

            y_emotions_list.append(y_causes)
            y_causes_list_1.append(y_causes)
            doc_id_list.append(doc_id)
            doc_len_list.append(doc_len)
            doc_couples_list.append(doc_couples)
            #print('third is',doc_id,y_causes)
            y_causes_list.append(2)  # find cause
            #print(y_emotions)
            negative = []
            
            # find cause=specific query
            for i,j in enumerate(y_causes):
                if j==1:
                    c = zeros.copy()
                    question = '[CLS]' + '那个是原因子句' + doc_clauses[i]['clause'] + '对应的情感子句'+ '[SEP] '
                    question += document
                    for k,z in enumerate(doc_causes):
                        if z== (i+1):
                            c[doc_emotions[k]-1] = 1
                        #print(c)                 
                    indexed_tokens = self.bert_tokenizer.encode(question.strip(), add_special_tokens=False)

                    clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                    clause_indices.remove(0)
                    doc_token_len = len(indexed_tokens)

                    segments_ids = []
                    segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                    segments_indices.append(len(indexed_tokens))
                    for i in range(len(segments_indices)-1):
                        semgent_len = segments_indices[i+1] - segments_indices[i]
                        if i % 2 == 0:
                            segments_ids.extend([0] * semgent_len)
                        else:
                            segments_ids.extend([1] * semgent_len)

                    #assert len(clause_indices) == doc_len
                    assert len(c) == doc_len
                    assert len(segments_ids) == len(indexed_tokens)
                    bert_token_idx_list.append(indexed_tokens)
                    bert_clause_idx_list.append(clause_indices)
                    bert_segments_idx_list.append(segments_ids)
                    bert_token_lens_list.append(doc_token_len)
                    
                    doc_id_list.append(doc_id)
                    doc_len_list.append(doc_len)
                    y_causes_list_1.append(y_causes)
                    y_emotions_list.append(c)
                    y_causes_list.append(3)  # find emotion
                    #print('here is',doc_id,c)
                    doc_couples_list.append(doc_couples)
                else:
                    negative.append(i)
                    
                    
            # negative sampling
            if torch.rand(1) < 0.1:
                negative = random.sample(negative, 1)
                c = zeros.copy()
                question = '[CLS]' + '那个是原因子句' + doc_clauses[negative[0]]['clause'] + '对应的情感子句'+ '[SEP] '
                question += document

                indexed_tokens = self.bert_tokenizer.encode(question.strip(), add_special_tokens=False)

                clause_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                clause_indices.remove(0)
                doc_token_len = len(indexed_tokens)

                segments_ids = []
                segments_indices = [i for i, x in enumerate(indexed_tokens) if x == 101]
                segments_indices.append(len(indexed_tokens))
                for i in range(len(segments_indices)-1):
                    semgent_len = segments_indices[i+1] - segments_indices[i]
                    if i % 2 == 0:
                        segments_ids.extend([0] * semgent_len)
                    else:
                        segments_ids.extend([1] * semgent_len)

                #assert len(clause_indices) == doc_len
                assert len(c) == doc_len
                assert len(segments_ids) == len(indexed_tokens)
                bert_token_idx_list.append(indexed_tokens)
                bert_clause_idx_list.append(clause_indices)
                bert_segments_idx_list.append(segments_ids)
                bert_token_lens_list.append(doc_token_len)

                doc_id_list.append(doc_id)
                doc_len_list.append(doc_len)
                doc_couples_list.append(doc_couples)

                #print('second is',c,doc_id)
                y_causes_list_1.append(y_causes)
                y_emotions_list.append(c)
                y_causes_list.append(3)  # find cause

        return doc_couples_list, y_emotions_list, y_causes_list, doc_len_list, doc_id_list, \
               bert_token_idx_list, bert_clause_idx_list, bert_segments_idx_list, bert_token_lens_list,y_causes_list_1

    def token_trunk(self, bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens,
                    doc_couples, y_emotions, y_causes, doc_len,y_causes_list_1):
        # TODO: cannot handle some extreme cases now
        #print('before',doc_couples,doc_len,bert_token_idx,bert_clause_idx,bert_token_lens)
        #print('before',bert_token_idx.index(102)+1,bert_clause_idx[0])
        emotion, cause = doc_couples[0]
        if emotion > doc_len / 2 and cause >= doc_len / 2:
            i = 0
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_token_idx.index(102)+1] + bert_token_idx[bert_clause_idx[i]:]
                if len(temp_bert_token_idx[temp_bert_token_idx.index(102)+1:]) <= 400:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:bert_token_idx.index(102)+1] + bert_token_idx[cls_idx:]
                    bert_segments_idx = [1-bert_segments_idx[cls_idx]] * bert_clause_idx[0] + bert_segments_idx[cls_idx:]
                    #bert_segments_idx = bert_segments_idx[:bert_clause_idx[0]] + bert_segments_idx[cls_idx:] # some problem, fix 
                    bert_clause_idx = [p - cls_idx for p in bert_clause_idx[i:]]
                    doc_couples = [[emotion - i, cause - i]]
                    y_emotions = y_emotions[i:]
                    y_causes_list_1 = y_causes_list_1[i:]
                    y_causes = y_causes
                    doc_len = doc_len - i
                    break
                i = i + 1
        if emotion < doc_len / 2 and cause < doc_len / 2:
            i = doc_len - 1
            while True:
                temp_bert_token_idx = bert_token_idx[:bert_clause_idx[i]]
                if len(temp_bert_token_idx[temp_bert_token_idx.index(102)+1:]) <= 400:
                    cls_idx = bert_clause_idx[i]
                    bert_token_idx = bert_token_idx[:cls_idx]
                    bert_segments_idx = bert_segments_idx[:cls_idx]
                    #bert_segments_idx = bert_segments_idx[:bert_clause_idx[0]] + bert_segments_idx[cls_idx:] # some problem, fix
                    bert_clause_idx = bert_clause_idx[:i]
                    y_emotions = y_emotions[:i]
                    y_causes_list_1 = y_causes_list_1[:i]
                    y_causes = y_causes
                    doc_len = i
                    break
                i = i - 1

        #print('after trunk',len(bert_token_idx))
        return bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
               doc_couples, y_emotions, y_causes, doc_len,y_causes_list_1


    def get_pair(self, data_type):
        if data_type == 'train':
            data_file = self.train_file
        elif data_type == 'valid':
            data_file = self.valid_file
        elif data_type == 'test':
            data_file = self.test_file
        data_list = read_json(data_file)
        pairs = []
        for doc in data_list:
            doc_id = doc['doc_id']
            doc_len = doc['doc_len']
            doc_couples = doc['pairs']
            doc_couples.sort()
            pair_single = []
            for a in doc_couples:
                pairs.append(int(doc_id) * 10000 + a[0] * 100 + a[1])
            #pairs.extend([pair_single])
        return pairs

    def get_pair1(self):
        pairs = []
        doc = []
        for idx in range(len(self.doc_couples_list)):
            doc_couples, y_emotions, y_causes = self.doc_couples_list[idx], self.y_emotions_list[idx], self.y_causes_list[idx]
            doc_len, doc_id = self.doc_len_list[idx], self.doc_id_list[idx]
            bert_token_idx, bert_clause_idx = self.bert_token_idx_list[idx], self.bert_clause_idx_list[idx]
            bert_segments_idx, bert_token_lens = self.bert_segments_idx_list[idx], self.bert_token_lens_list[idx]
            y_causes_list_1 = self.y_causes_list_1[idx]
            
            if doc_id in doc:
                continue

            if len(bert_token_idx[bert_token_idx.index(102)+1:]) > 400 and doc_id not in doc:
                #print(doc_id,'need trunk')
                bert_token_idx, bert_clause_idx, bert_segments_idx, bert_token_lens, \
                doc_couples, y_emotions, y_causes, doc_len,y_causes_list_1 = self.token_trunk(bert_token_idx, bert_clause_idx,
                                                                              bert_segments_idx, bert_token_lens,
                                                                              doc_couples, y_emotions, y_causes, doc_len,y_causes_list_1)
            doc.append(doc_id)
            #print(doc_id,doc_couples)         
            for a in doc_couples:
                pairs.append(int(doc_id) * 10000 + a[0] * 100 + a[1])

        return list(set(pairs))

def bert_batch_preprocessing(batch):
    doc_couples_b, y_emotions_b, y_causes_b, doc_len_b, doc_id_b, \
    bert_token_b, bert_segment_b, bert_clause_b, bert_token_lens_b,y_causes_list_1 = zip(*batch)

    y_mask_b, y_emotions_b = pad_docs(doc_len_b, y_emotions_b)
    _, y_causes_list_1 = pad_docs(doc_len_b, y_causes_list_1)
    adj_b = pad_matrices(doc_len_b)
    bert_token_b = pad_sequence(bert_token_b, batch_first=True, padding_value=0)
    bert_segment_b = pad_sequence(bert_segment_b, batch_first=True, padding_value=0)
    bert_clause_b = pad_sequence(bert_clause_b, batch_first=True, padding_value=0)

    bsz, max_len = bert_token_b.size()
    bert_masks_b = np.zeros([bsz, max_len], dtype=np.float)
    for index, seq_len in enumerate(bert_token_lens_b):
        bert_masks_b[index][:seq_len] = 1

    bert_masks_b = torch.FloatTensor(bert_masks_b)
    #print(bert_segment_b.shape,bert_token_b.shape)
    assert bert_segment_b.shape == bert_token_b.shape
    assert bert_segment_b.shape == bert_masks_b.shape

    return np.array(doc_len_b), np.array(adj_b), \
           np.array(y_emotions_b), np.array(y_causes_b), np.array(y_mask_b), doc_couples_b, doc_id_b, \
           bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b,np.array(y_causes_list_1)


def pad_docs(doc_len_b, y_emotions_b):
    #print('before',doc_len_b, y_emotions_b, y_causes_b)
    max_doc_len = max(doc_len_b)

    y_mask_b, y_emotions_b_ = [], []
    for y_emotions in y_emotions_b:
        y_emotions_ = pad_list(y_emotions, max_doc_len, -1)
        y_mask = list(map(lambda x: 0 if x == -1 else 1, y_emotions_))

        y_mask_b.append(y_mask)
        y_emotions_b_.append(y_emotions_)

    return y_mask_b, y_emotions_b_

def pad_matrices(doc_len_b):
    N = max(doc_len_b)
    adj_b = []
    for doc_len in doc_len_b:
        adj = np.ones((doc_len, doc_len))
        adj = sp.coo_matrix(adj)
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)),
                            shape=(N, N), dtype=np.float32)
        adj_b.append(adj.toarray())
    return adj_b


def pad_list(element_list, max_len, pad_mark):
    element_list_pad = element_list[:]
    pad_mark_list = [pad_mark] * (max_len - len(element_list))
    element_list_pad.extend(pad_mark_list)
    return element_list_pad

