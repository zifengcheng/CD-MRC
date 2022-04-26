#-*-coding:GBK -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE
from transformers import BertModel


class Network(nn.Module):
    def __init__(self, configs):
        super(Network, self).__init__()
        self.bert = BertModel.from_pretrained(configs.bert_cache_path)
        self.fc = nn.Linear(400,2)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768,200)

        self.lstm1 = nn.TransformerEncoderLayer(d_model=200, nhead=1,dim_feedforward=200, dropout=0)

        self.fc5 = nn.Linear(768,1)

    def forward(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len,y_causes_b):

        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),
                                attention_mask=bert_masks_b.to(DEVICE))
        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        doc_sents_h = self.lstm1(self.fc1(doc_sents_h).permute(1,0,2)).permute(1,0,2)
        doc_sents_h = torch.cat((doc_sents_h[:,0:1,:].expand_as(doc_sents_h[:,1:,:]),doc_sents_h[:,1:,:]),2)
        pred = self.fc(doc_sents_h)
    
        return pred

    '''def batched_index_select(self, bert_output, bert_clause_b):
        #print(bert_output)
        hidden_state = bert_output[0]
        print(hidden_state.shape,bert_clause_b.shape)
        dummy = bert_clause_b.unsqueeze(2).expand(bert_clause_b.size(0), bert_clause_b.size(1), hidden_state.size(2))
        print(dummy.shape,dummy)
        doc_sents_h = hidden_state.gather(1, dummy)
        print(doc_sents_h.shape)
        return doc_sents_h

    def batched_index_select(self, bert_output, bert_clause_b):
        #print(bert_output)
        hidden_state = bert_output[0]
        doc_sents_h = torch.zeros(bert_clause_b.size(0), bert_clause_b.size(1)+1, hidden_state.size(2)).cuda()
        #print(hidden_state.shape,bert_clause_b.shape)
        for i in range(doc_sents_h.shape[0]):
            for j in range(doc_sents_h.shape[1]):
                if j == doc_sents_h.shape[1] -1:
                    hidden = hidden_state[i,bert_clause_b[i,j-1]:,:]
                    weight = F.softmax(self.fc5(hidden),0)
                    hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden 
                elif bert_clause_b[i,j]!=0:
                    if j==0:
                        hidden = hidden_state[i,0:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    else:
                        hidden = hidden_state[i,bert_clause_b[i,j-1]:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden 
                
        return doc_sents_h'''
    
    def batched_index_select(self, bert_output, bert_clause_b):
        #print(bert_output)
        hidden_state = bert_output[0]
        doc_sents_h = torch.zeros(bert_clause_b.size(0), bert_clause_b.size(1) + 1, hidden_state.size(2)).cuda()
        #print(doc_sents_h.shape)
        #print(hidden_state.shape,bert_clause_b.shape)
        for i in range(doc_sents_h.shape[0]):
            for j in range(doc_sents_h.shape[1]):
                if j == doc_sents_h.shape[1] -1:
                    hidden = hidden_state[i,bert_clause_b[i,j-1]:,:]
                    weight = F.softmax(self.fc5(hidden),0)
                    hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                elif bert_clause_b[i,j]!=0:
                    if j==0:
                        hidden = hidden_state[i,0:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    else:
                        hidden = hidden_state[i,bert_clause_b[i,j-1]:bert_clause_b[i,j],:]
                        weight = F.softmax(self.fc5(hidden),0)
                        hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                else:
                    hidden = hidden_state[i,bert_clause_b[i,j-1]:,:]
                    weight = F.softmax(self.fc5(hidden),0)
                    hidden = torch.mm(hidden.permute(1,0),weight).squeeze(1)
                    doc_sents_h[i,j,:] = hidden
                    break    
            
        return doc_sents_h

    def loss_pre(self, pred_e, y_emotions, source_length):
        #print('loss function shape is ',pred_e.shape,y_emotions.shape,source_length)   #seq_len * batch  * class  .  batch * seq_len
        y_emotions = torch.LongTensor(y_emotions).to(DEVICE)
        packed_y = torch.nn.utils.rnn.pack_padded_sequence(pred_e.permute(1,0,2), list(source_length),enforce_sorted=False).data
        target_ = torch.nn.utils.rnn.pack_padded_sequence(y_emotions.permute(1,0), list(source_length),enforce_sorted=False).data
        loss_e  = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(packed_y), target_)
        return loss_e
    
    def inference(self, bert_token_b, bert_segment_b, bert_masks_b,
                bert_clause_b, doc_len,y_causes_b,doc_id):
        #print(y_causes_b,y_causes_b.shape)
        doc_ids = list(doc_id)
        doc_ids_2 = []
        for a in doc_ids:
            doc_ids_2.append(int(a))
 
        doc_id_2 = torch.masked_select(torch.tensor(doc_ids_2),(torch.from_numpy(y_causes_b) == 2))        
        y_causes_b_2 = (torch.from_numpy(y_causes_b)==2).unsqueeze(1)

        bert_token_b_ = torch.masked_select(bert_token_b,y_causes_b_2).view(-1,bert_masks_b.shape[1])
        bert_masks_b_ = torch.masked_select(bert_masks_b,y_causes_b_2).view(-1,bert_token_b.shape[1])
        bert_segment_b_ = torch.masked_select(bert_segment_b,y_causes_b_2).view(-1,bert_segment_b.shape[1])
        bert_clause_b_ = torch.masked_select(bert_clause_b,y_causes_b_2).view(bert_masks_b_.shape[0],-1)
                             
        doc_id = list(doc_id)
        doc_ids = []
        for a in doc_id:
            doc_ids.append(int(a))

        doc_id = torch.masked_select(torch.tensor(doc_ids),(1-torch.from_numpy(y_causes_b).bool().long()).bool())

        #print(doc_id==doc_id_2)
        
        y_causes_b = (1-torch.from_numpy(y_causes_b).bool().long()).unsqueeze(1).bool()
        
        bert_token_b = torch.masked_select(bert_token_b,y_causes_b).view(-1,bert_masks_b.shape[1])
        bert_masks_b = torch.masked_select(bert_masks_b,y_causes_b).view(-1,bert_token_b.shape[1])
        bert_segment_b = torch.masked_select(bert_segment_b,y_causes_b).view(-1,bert_segment_b.shape[1])
        bert_clause_b = torch.masked_select(bert_clause_b,y_causes_b).view(bert_masks_b.shape[0],-1)
        
        bert_output = self.bert(input_ids=bert_token_b.to(DEVICE),attention_mask=bert_masks_b.to(DEVICE))

        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b.to(DEVICE))
        doc_sents_h = self.lstm1(self.fc1(doc_sents_h.permute(1,0,2))).permute(1,0,2)
        doc_sents_h = torch.cat((doc_sents_h[:,0:1,:].expand_as(doc_sents_h[:,1:,:]),doc_sents_h[:,1:,:]),2)
        pred_e = self.fc(doc_sents_h).argmax(2)
                                                                                                                                                                                
        # construct input
        pair1,pair2 = [],[]
        pair3,pair4 = [],[]
        pair5,pair6 = [],[]
        pair7,pair8 = [],[]
        pair9,pair10 = [],[]
        
        for i in range(pred_e.shape[0]): # batch 
            c = bert_token_b[i].numpy().tolist().copy()
            if 0 in c:
                document = c[c.index(101,1):c.index(0)]
            else:
                document = c[c.index(101,1):]
                
            b_clause_b = bert_clause_b[i].tolist()
            tmp = []
            for z in b_clause_b:
                if z !=0:
                    tmp.append(z-9)
                else:
                    tmp.append(0)

            for j in range(pred_e.shape[1]): # seq_len
                input = []
                if pred_e[i,j] == 1:
                    if tmp[j] ==0:
                        continue
                    elif pred_e.shape[1]-1 ==j or tmp[j+1]==0: # final clause
                        emotion_cause = [101,6929,702,3221,2658,2697,2094,1368] + document[tmp[j]+1:-1] + [2190,2418,4638,1333,1728,2094,1368,102]
                    else:
                        emotion_cause = [101,6929,702,3221,2658,2697,2094,1368] + document[tmp[j]+1:tmp[j+1]-1] + [2190,2418,4638,1333,1728,2094,1368,102]
                    #print('emotion_cause is ',emotion_cause)

                    input = emotion_cause + document

                    #print(input)
                    bert_clause_b_1 = [i for i, x in enumerate(input) if x == 101]
                    bert_clause_b_1.remove(0)
                    bert_clause_b_2 = torch.tensor([bert_clause_b_1])
                    input_ids = torch.tensor([input])
                    
                    segments_ids = []
                    segments_indices = [k for k, x in enumerate(input) if x == 101]
                    segments_indices.append(len(input))
                    for k in range(len(segments_indices)-1):
                        semgent_len = segments_indices[k+1] - segments_indices[k]
                        if k % 2 == 0:
                            segments_ids.extend([0] * semgent_len)
                        else:
                            segments_ids.extend([1] * semgent_len)
                    bert_segment_b = torch.tensor([segments_ids])
                    #print(input,segments_ids)
                    #print(input_ids.shape)
                    if input_ids.shape[1]>512:
                        print(doc_id[i])
                        continue
                    
                    bert_output = self.bert(input_ids=input_ids.to(DEVICE),
                                            attention_mask = input_ids.bool().long().to(DEVICE))
                    #print(bert_output[0].shape,bert_output.last_hidden_state.shape)
                    doc_sents_h = self.batched_index_select(bert_output, bert_clause_b_2.to(DEVICE))
                    doc_sents_h = self.lstm1(self.fc1(doc_sents_h).permute(1,0,2)).permute(1,0,2)
                    doc_sents_h = torch.cat((doc_sents_h[:,0:1,:].expand_as(doc_sents_h[:,1:,:]),doc_sents_h[:,1:,:]),2)
                    pred_c = F.softmax(self.fc(doc_sents_h),-1).squeeze(0)
                    for k in range(pred_c.shape[0]):
                        if pred_c[k,1] >=0.5:
                            pair1.append(int(doc_id[i] * 10000 +j * 100 + k+101))
                            if pred_c[k,1] >=0.6:
                                pair3.append(int(doc_id[i] * 10000 +j * 100 + k+101))
                                if pred_c[k,1] >=0.7:
                                    pair5.append(int(doc_id[i] * 10000 +j * 100 + k+101))
                                    if pred_c[k,1] >=0.8:
                                        pair7.append(int(doc_id[i] * 10000 +j * 100 + k+101))
                                        if pred_c[k,1] >=0.9:
                                            pair9.append(int(doc_id[i] * 10000 +j * 100 + k+101))


                            
                            
        #print('finish')                                 
        #print('pair1 is',pair1)                
        
        # pair extraction based on cause-guided                         

        #print('after filter',bert_token_b.shape,bert_masks_b.shape)
        bert_output = self.bert(input_ids=bert_token_b_.to(DEVICE),attention_mask=bert_masks_b_.to(DEVICE))

        doc_sents_h = self.batched_index_select(bert_output, bert_clause_b_.to(DEVICE))
        doc_sents_h = self.lstm1(self.fc1(doc_sents_h).permute(1,0,2)).permute(1,0,2)
        doc_sents_h = torch.cat((doc_sents_h[:,0:1,:].expand_as(doc_sents_h[:,1:,:]),doc_sents_h[:,1:,:]),2)
        pred_c = self.fc(doc_sents_h).argmax(2)
                                                                                                                                                                                
        # construct input
        
        for i in range(pred_c.shape[0]): # batch 
            c = bert_token_b_[i].numpy().tolist().copy()
            if 0 in c:
                document = c[c.index(101,1):c.index(0)]
            else:
                document = c[c.index(101,1):]
                
            b_clause_b = bert_clause_b_[i].tolist()
            #b_clause_b.pop(0)
            #b_clause_b.append(0)
            #b_clause_b.insert(0,0)
            tmp = []
            for z in b_clause_b:
                if z !=0:
                    tmp.append(z-9)
                else:
                    tmp.append(0)

                    
            #print('document is ',document,doc_id[i],tmp)
            for j in range(pred_c.shape[1]): # seq_len
                input = []
                #print(pred_e.shape)
                if pred_c[i,j] == 1:
                    #print('document',doc_id_2[i],'clause', j ,'is cause clause')
                    #print('document is ',document)
                    #print(j,pred_e.shape)
                    #if bert_clause_b[]
                    if tmp[j] ==0:
                        continue
                    elif pred_c.shape[1]-1 ==j or tmp[j+1]==0: # final clause
                        emotion_cause = [101,6929,702,3221,1333,1728,2094,1368] + document[tmp[j]+1:-1] + [2190,2418,4638,2658,2697,2094,1368,102]
                    else:
                        emotion_cause = [101,6929,702,3221,1333,1728,2094,1368] + document[tmp[j]+1:tmp[j+1]-1] + [2190,2418,4638,2658,2697,2094,1368,102]                    

                    input = emotion_cause + document

                    #print(input)
                    bert_clause_b_1 = [i for i, x in enumerate(input) if x == 101]
                    bert_clause_b_1.remove(0)
                    bert_clause_b_2 = torch.tensor([bert_clause_b_1])
                    input_ids = torch.tensor([input])
                    
                    segments_ids = []
                    segments_indices = [k for k, x in enumerate(input) if x == 101]
                    segments_indices.append(len(input))
                    for k in range(len(segments_indices)-1):
                        semgent_len = segments_indices[k+1] - segments_indices[k]
                        if k % 2 == 0:
                            segments_ids.extend([0] * semgent_len)
                        else:
                            segments_ids.extend([1] * semgent_len)
                    bert_segment_b = torch.tensor([segments_ids])
                    #print(input,segments_ids)
                    #print(input_ids.shape)
                    if input_ids.shape[1]>512:
                        print(doc_id_2[i])
                        continue
                    
                    bert_output = self.bert(input_ids=input_ids.to(DEVICE),
                                            attention_mask = input_ids.bool().long().to(DEVICE))

                    #print(bert_output[0].shape,bert_output.last_hidden_state.shape)
                    doc_sents_h = self.batched_index_select(bert_output, bert_clause_b_2.to(DEVICE))
                    doc_sents_h = self.lstm1(self.fc1(doc_sents_h).permute(1,0,2)).permute(1,0,2)
                    doc_sents_h = torch.cat((doc_sents_h[:,0:1,:].expand_as(doc_sents_h[:,1:,:]),doc_sents_h[:,1:,:]),2)
                    pred_c_ = F.softmax(self.fc(doc_sents_h),-1).squeeze(0)
                    for k in range(pred_c_.shape[0]):
                        if pred_c_[k,1] >=0.5:
                            pair2.append(int(doc_id[i] * 10000 +k * 100 + j+101))
                            if pred_c_[k,1] >=0.6:
                                pair4.append(int(doc_id[i] * 10000 +k * 100 + j+101))
                                if pred_c_[k,1] >=0.7:
                                    pair6.append(int(doc_id[i] * 10000 +k * 100 + j+101))
                                    if pred_c_[k,1] >=0.8:
                                        pair8.append(int(doc_id[i] * 10000 +k * 100 + j+101))
                                        if pred_c_[k,1] >=0.9:
                                            pair10.append(int(doc_id[i] * 10000 +k * 100 + j+101))


        return pair1,pair2,pred_e,pred_c,pair3,pair4,pair5,pair6,pair7,pair8,pair9,pair10

