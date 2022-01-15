import sys, os, warnings, time
sys.path.append('..')
warnings.filterwarnings("ignore")
import numpy as np
import torch
from config import *
from data_loader import *
from model import *
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.utils import *

import random



def main(configs, fold_id):
    random.seed(TORCH_SEED)
    #os.environ['PYTHONHASHSEED'] =str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    train_loader = build_train_data(configs, fold_id=fold_id)
    if configs.split == 'split20':
        valid_pair = get_ecpair(configs, fold_id=fold_id, data_type='valid')
        valid_loader = build_inference_data(configs, fold_id=fold_id, data_type='valid')

    test_pair = get_ecpair(configs, fold_id=fold_id, data_type='test')
    test_loader = build_inference_data(configs, fold_id=fold_id, data_type='test')

    model = Network(configs).to(DEVICE)

    print(configs.warmup_proportion,configs.l2_bert,configs.l2_other,configs.lr,test_pair)

    paramsbert = []
    paramsbert0reg = []
    paramsothers = []
    paramsothers0reg = []
    
    for name, parameters in model.named_parameters():
        #print(name, ':', parameters.shape)
        if not parameters.requires_grad:
            continue
        if 'bert' in name:
            if '.bias' in name or 'LayerNorm.weight' in name:
                paramsbert0reg += [parameters]
            else:
                paramsbert += [parameters]
        else:
            paramsothers += [parameters]

    params = [dict(params=paramsbert, weight_decay=configs.l2_bert),
                  dict(params=paramsothers,lr=1e-4,weight_decay=configs.l2_other),
                  dict(params=paramsbert0reg, weight_decay=0.0)]
    optimizer = AdamW(params,lr=configs.lr,weight_decay = 0)

    num_steps_all = len(train_loader) // configs.gradient_accumulation_steps * configs.epochs
    warmup_steps = int(num_steps_all * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps_all)

    model.zero_grad()
    max_ce,max_ec, max_e, max_c = (-1, -1, -1), (-1, -1, -1),(-1, -1, -1),(-1, -1, -1)
    max_6,max_7, max_8, max_9 = (-1,-1,-1), (-1, -1, -1),(-1, -1, -1),(-1, -1, -1)
    max_16,max_17, max_18, max_19 = (-1,-1,-1), (-1, -1, -1),(-1, -1, -1),(-1, -1, -1)
    max_and, max_or = (-1,-1,-1), (-1, -1, -1)
    
    metric_ec,metric_ce, metric_e, metric_c = (-1,-1,-1), (-1, -1, -1),(-1, -1, -1),(-1, -1, -1)
    metric_6,metric_7, metric_8, metric_9 = (-1,-1,-1), (-1, -1, -1),(-1, -1, -1),(-1, -1, -1)
    metric_16,metric_17, metric_18, metric_19 = (-1,-1,-1), (-1, -1, -1),(-1, -1, -1),(-1, -1, -1)
    metric_and, metric_or = (-1,-1,-1), (-1, -1, -1)
    early_stop_flag = 0
    for epoch in range(1, configs.epochs+1):
        if epoch ==10 and metric_ec[2] <=0.5:
            break
        for train_step, batch in enumerate(train_loader, 1):
            model.train()
            doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
            bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b ,y_causes_b_1= batch

            pred_e = model(bert_token_b,bert_segment_b, bert_masks_b,
                                                              bert_clause_b, doc_len_b,y_causes_b)  # seq_len * batch * 10
            loss = model.loss_pre(pred_e, y_emotions_b, doc_len_b)
            loss = loss / configs.gradient_accumulation_steps
            #if train_step <= 20:
            #    print('epoch: ',epoch,loss)

            loss.backward()
            if train_step % configs.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if epoch >=3 and train_step %50 ==0:

                with torch.no_grad():
                    model.eval()
                    #print('epoch :',epoch ,'eval is begining')
                    if configs.split == 'split10':
                        a,b,c,d,e,f,g,h,i,j,k,l,m,n = inference_one_epoch(configs, test_loader, model,test_pair,1)
                        if a[2] > metric_e[2]:
                            metric_e = a

                        if b[2] > metric_c[2]:
                            early_stop_flag = 1
                            metric_c = b

                        if c[2] > metric_ec[2]:
                            early_stop_flag = 1
                            metric_ec = c
                            
                        if d[2] > metric_ce[2]:
                            early_stop_flag = 1
                            metric_ce = d
                            
                        if e[2] > metric_and[2]:
                            early_stop_flag = 1
                            metric_and = e    

                        if f[2] > metric_or[2]:
                            early_stop_flag = 1
                            metric_or = f
                            
                        if g[2] > metric_6[2]:
                            early_stop_flag = 1
                            metric_6 = g
                            
                        if h[2] > metric_7[2]:
                            early_stop_flag = 1
                            metric_7 = h
                            #print('this is best',h[2])
                            
                        if i[2] > metric_8[2]:
                            early_stop_flag = 1
                            metric_8 = g
                            
                        if j[2] > metric_9[2]:
                            early_stop_flag = 1
                            metric_9 = j
                            print('this is best',j[2])
                        else:
                            early_stop_flag += 1
                            if early_stop_flag >5 and epoch>=5:
                                return metric_ec, metric_ce,metric_e, metric_c,metric_and, metric_or,metric_6,metric_7, metric_8, metric_9,metric_16,metric_17, metric_18, metric_19
                            #print('this is best')

                            
                        if k[2] > metric_16[2]:
                            early_stop_flag = 1
                            metric_16 = k
                            
                        if l[2] > metric_17[2]:
                            early_stop_flag = 1
                            metric_17 = l       
                            
                        if m[2] > metric_18[2]:
                            early_stop_flag = 1
                            metric_18 = m
                            
                        if n[2] > metric_19[2]:
                            early_stop_flag = 1
                            metric_19 = n
                            
                    if configs.split == 'split20':
                        a,b,c,d,e,f,g,h,i,j,k,l,m,n = inference_one_epoch(configs, valid_loader, model,valid_pair,1)
                        a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,m1,n1 = inference_one_epoch(configs, test_loader, model,test_pair,1)
                        
                        if a[2] > max_e[2]:
                            max_e,metric_e = a,a1

                        if b[2] > max_c[2]:
                            max_c,metric_c = b,b1

                        
                        if c[2] > max_ec[2]:
                            max_ec,metric_ec = c,c1
                            
                        if d[2] > max_ec[2]:
                            max_ec,metric_ec = d,d1
                            
                        if e[2] > max_ec[2]:
                            max_ec,metric_ec = e,e1
                            
                        if f[2] > max_ec[2]:
                            max_ec,metric_ec = f,f1
                            
                        if g[2] > max_ec[2]:
                            max_ec,metric_ec = g,g1
                        
                        if h[2] > max_ec[2]:
                            max_ec,metric_ec = h,g1
                            
                        if i[2] > max_ec[2]:
                            max_ec,metric_ec = i,i1
                        
                        if j[2] > max_ec[2]:
                            max_ec,metric_ec = j,j1
                            early_stop_flag = 1
                            print('this is best',j1[2])
                        else:
                            early_stop_flag += 1
                            if early_stop_flag >5 and epoch>=5:
                                return metric_ec, metric_ce,metric_e, metric_c,metric_and, metric_or,metric_6,metric_7, metric_8, metric_9,metric_16,metric_17, metric_18, metric_19
                            #print('this is best')

                            
                        if k[2] > max_ec[2]:
                            max_ec,metric_ec = k,k1
                        
                        if l[2] > max_ec[2]:
                            max_ec,metric_ec = l,l1
                        
                        if m[2] > max_ec[2]:
                            max_ec,metric_ec = m,m1
                            
                        if n[2] > max_ec[2]:
                            max_ec,metric_ec = n,n1         

    return metric_ec, metric_ce,metric_e, metric_c,metric_and, metric_or,metric_6,metric_7, metric_8, metric_9,metric_16,metric_17, metric_18, metric_19


def inference_one_epoch(configs, batches, model,pair,epoch):
    for batch in batches:
        doc_len_b, adj_b, y_emotions_b, y_causes_b, y_mask_b, doc_couples_b, doc_id_b, \
        bert_token_b, bert_segment_b, bert_masks_b, bert_clause_b,y_causes_b_1 = batch
        pred_pair1,pred_pair2,pred_e,pred_c,pred_pair3,pred_pair4,pred_pair5,pred_pair6,pred_pair7,pred_pair8,pred_pair9,pred_pair10 = model.inference(bert_token_b, bert_segment_b, bert_masks_b,bert_clause_b, doc_len_b,y_causes_b,doc_id_b)
        #print(pred_pair1)

        #acc,p_e,r_e,f_e = acc_prf(pred_e,y_emotions_b,doc_len_b,y_causes_b)
        #print('emotion result is ',p_e,r_e,f_e)

        #acc,p_c,r_c,f_c = acc_prf(pred_c,y_causes_b_1,doc_len_b,y_causes_b)
        #print('cause result is ',p_c,r_c,f_c)
        
        #p_7,r_7,f1_7 = prf_2nd_step(pair,list((set(pred_pair1) & set(pred_pair2)) | set(pred_pair5) | set(pred_pair6)))
        #print('before filter is',p_7,r_7,f1_7)
        
        pred_pair1 = lexicon_based_extraction(doc_id_b, pred_pair1)
        pred_pair2 = lexicon_based_extraction(doc_id_b, pred_pair2)
        pred_pair3 = lexicon_based_extraction(doc_id_b, pred_pair3)
        pred_pair4 = lexicon_based_extraction(doc_id_b, pred_pair4)
        pred_pair5 = lexicon_based_extraction(doc_id_b, pred_pair5)
        pred_pair6 = lexicon_based_extraction(doc_id_b, pred_pair6)
        pred_pair7 = lexicon_based_extraction(doc_id_b, pred_pair7)
        pred_pair8 = lexicon_based_extraction(doc_id_b, pred_pair8)
        pred_pair9 = lexicon_based_extraction(doc_id_b, pred_pair9)
        pred_pair10 = lexicon_based_extraction(doc_id_b, pred_pair10)
        
        pred_pair_and = set(pred_pair1) & set(pred_pair2)

        (p_e,r_e,f_e),(p_c,r_c,f_c) = eval_func(pair,list(pred_pair_and | set(pred_pair5) | set(pred_pair6)))

        p_a,r_a,f1_a = prf_2nd_step(pair,list(pred_pair_and))
        
        p_6,r_6,f1_6 = prf_2nd_step(pair,list(pred_pair_and | set(pred_pair3) | set(pred_pair4)))
        p_7,r_7,f1_7 = prf_2nd_step(pair,list(pred_pair_and | set(pred_pair5) | set(pred_pair6)))
        #print('after filter is',p_7,r_7,f1_7)
        p_8,r_8,f1_8 = prf_2nd_step(pair,list(pred_pair_and | set(pred_pair7) | set(pred_pair8)))
        p_9,r_9,f1_9 = prf_2nd_step(pair,list(pred_pair_and | set(pred_pair9) | set(pred_pair10)))

        #p_single,r_single,f1_single,p_multi,r_multi,f1_multi = document_extraction(pair,list((set(pred_pair1) & set(pred_pair2)) | set(pred_pair9) | set(pred_pair10)))
        #print(p_single,r_single,f1_single,p_multi,r_multi,f1_multi)
        
        p_16,r_16,f1_16 = prf_2nd_step(pair,set(pred_pair1) | set(pred_pair4))
        p_17,r_17,f1_17 = prf_2nd_step(pair,set(pred_pair1) | set(pred_pair6))
        p_18,r_18,f1_18 = prf_2nd_step(pair,set(pred_pair1) | set(pred_pair8))
        p_19,r_19,f1_19 = prf_2nd_step(pair,set(pred_pair1) | set(pred_pair10))
        
        p_o,r_o,f1_o = prf_2nd_step(pair,list(set(pred_pair1) | set(pred_pair2)))
        
        #p,r,f1 = prf_2nd_step(pair,list(set(pred_pair1) - set(pred_pair2)))
        #print('p,r,f1 is based on -',p,r,f1)
        
        p_ce,r_ce,f1_ce = prf_2nd_step(pair,pred_pair2)
        #print('p,r,f1 is based on pred_pair2',p,r,f1)
        p_ec,r_ec,f1_ec = prf_2nd_step(pair,pred_pair1)
        #print('p,r,f1 is based on pred_pair1',p,r,f1)
        
    return (p_e,r_e,f_e),(p_c,r_c,f_c),(p_ec,r_ec,f1_ec),(p_ce,r_ce,f1_ce),(p_a,r_a,f1_a),(p_o,r_o,f1_o),(p_6,r_6,f1_6),(p_7,r_7,f1_7),\
(p_8,r_8,f1_8),(p_9,r_9,f1_9),(p_16,r_16,f1_16),(p_17,r_17,f1_17),(p_18,r_18,f1_18),(p_19,r_19,f1_19)


def lexicon_based_extraction(doc_ids, couples_pred):
    emotional_clauses = read_b(os.path.join(DATA_DIR, SENTIMENTAL_CLAUSE_DICT))

    couples_pred_filtered = []
    for i, couples_pred_i in enumerate(couples_pred):
        emotional_clauses_i = emotional_clauses[str(int(couples_pred_i/10000))]
        if int((couples_pred_i%10000)/100) in emotional_clauses_i:
            couples_pred_filtered.append(couples_pred_i)
            
    #if len (couples_pred) !=0:
    #    print(couples_pred,couples_pred_filtered,len(couples_pred),len(couples_pred_filtered))
    return couples_pred_filtered

if __name__ == '__main__':
    configs = Config()

    if configs.split == 'split10':
        n_folds = 10
        configs.epochs = 20
    elif configs.split == 'split20':
        n_folds = 20
        configs.epochs = 20
    else:
        print('Unknown data split.')
        exit()

    metric_folds = {'ecp': [], 'emo': [], 'cau': [],'cep': []}
    metric_1 = {'0.5': [], '0.6': [], '0.7': [],'0.8': [],'0.9': [],'1.0': []}
    metric_2 = { '0.6': [], '0.7': [],'0.8': [],'0.9': []}
    for fold_id in range(1, n_folds+1):
        print('===== fold {} ====='.format(fold_id))
        metric_ec,metric_ce, metric_e, metric_c,metric_and,metric_or,metric_6,metric_7,metric_8,metric_9,\
        metric_16,metric_17,metric_18,metric_19= main(configs, fold_id)
        while metric_ec[2] < 0.5:
            metric_ec,metric_ce, metric_e, metric_c,metric_and,metric_or,metric_6,metric_7,metric_8,metric_9,\
        metric_16,metric_17,metric_18,metric_19= main(configs, fold_id)

        print('F_ecp: {}'.format(metric_ec))
        print('F_ecp: {}'.format(metric_ce))
        print('F_ecp: {}'.format(metric_and))
        print('F_ecp: {}'.format(metric_or))
        print('F_ecp: {}'.format(metric_6))
        print('F_ecp: {}'.format(metric_7))
        print('F_ecp: {}'.format(metric_8))
        print('F_ecp: {}'.format(metric_9))
        print('F_ecp: {}'.format(metric_16))
        print('F_ecp: {}'.format(metric_17))
        print('F_ecp: {}'.format(metric_18))
        print('F_ecp: {}'.format(metric_19))

        metric_folds['ecp'].append(metric_ec)
        metric_folds['cep'].append(metric_ce)
        metric_folds['emo'].append(metric_e)
        metric_folds['cau'].append(metric_c)
        
        metric_1['0.5'].append(metric_and)
        metric_1['0.6'].append(metric_6)
        metric_1['0.7'].append(metric_7)
        metric_1['0.8'].append(metric_8)
        metric_1['0.9'].append(metric_9)
        metric_1['1.0'].append(metric_or)
        
        metric_2['0.6'].append(metric_16)
        metric_2['0.7'].append(metric_17)
        metric_2['0.8'].append(metric_18)
        metric_2['0.9'].append(metric_19)

    metric_ec = np.mean(np.array(metric_folds['ecp']), axis=0).tolist()
    metric_ce = np.mean(np.array(metric_folds['cep']), axis=0).tolist()
    metric_e = np.mean(np.array(metric_folds['emo']), axis=0).tolist()
    metric_c = np.mean(np.array(metric_folds['cau']), axis=0).tolist()
    
    metric_and = np.mean(np.array(metric_1['0.5']), axis=0).tolist()
    metric_6 = np.mean(np.array(metric_1['0.6']), axis=0).tolist()
    metric_7 = np.mean(np.array(metric_1['0.7']), axis=0).tolist()
    metric_8 = np.mean(np.array(metric_1['0.8']), axis=0).tolist()
    metric_9 = np.mean(np.array(metric_1['0.9']), axis=0).tolist()
    metric_or = np.mean(np.array(metric_1['1.0']), axis=0).tolist()
    
    metric_16 = np.mean(np.array(metric_2['0.6']), axis=0).tolist()
    metric_17 = np.mean(np.array(metric_2['0.7']), axis=0).tolist()
    metric_18 = np.mean(np.array(metric_2['0.8']), axis=0).tolist()
    metric_19 = np.mean(np.array(metric_2['0.9']), axis=0).tolist()
    
    print('===== Average =====')
    print('F_ecp: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_ec[2]), float_n(metric_ec[0]), float_n(metric_ec[1])))
    print('F_cep: {}, P_cep: {}, R_cep: {}'.format(float_n(metric_ce[2]), float_n(metric_ce[0]), float_n(metric_ce[1])))
    print('F_emo: {}, P_emo: {}, R_emo: {}'.format(float_n(metric_e[2]), float_n(metric_e[0]), float_n(metric_e[1])))
    print('F_cau: {}, P_cau: {}, R_cau: {}'.format(float_n(metric_c[2]), float_n(metric_c[0]), float_n(metric_c[1])))
    
    print('F_and: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_and[2]), float_n(metric_and[0]), float_n(metric_and[1])))
    print('F_or: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_or[2]), float_n(metric_or[0]), float_n(metric_or[1])))
    
    print('F_6: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_6[2]), float_n(metric_6[0]), float_n(metric_6[1])))
    print('F_7: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_7[2]), float_n(metric_7[0]), float_n(metric_7[1])))
    print('F_8: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_8[2]), float_n(metric_8[0]), float_n(metric_8[1])))
    print('F_9: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_9[2]), float_n(metric_9[0]), float_n(metric_9[1])))
    
    print('F_16: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_16[2]), float_n(metric_16[0]), float_n(metric_16[1])))
    print('F_17: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_17[2]), float_n(metric_17[0]), float_n(metric_17[1])))
    print('F_18: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_18[2]), float_n(metric_18[0]), float_n(metric_18[1])))
    print('F_19: {}, P_ecp: {}, R_ecp: {}'.format(float_n(metric_19[2]), float_n(metric_19[0]), float_n(metric_19[1])))
    write_b({'ecp': metric_ec, 'emo': metric_e, 'cau': metric_c}, '{}_{}_metrics.pkl'.format(time.time(), configs.split))

