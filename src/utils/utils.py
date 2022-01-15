import pickle, json, decimal, math,torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def document_extraction(doc_couples_all, doc_couples_pred_all):
    #print(doc_couples_all, doc_couples_pred_all)
    if len(doc_couples_pred_all) ==0:
        return 0,0,0,0,0,0
    doc_couples_all, doc_couples_pred_all = list(set(doc_couples_all)),list(set(doc_couples_pred_all))
    b,single = [] ,[]
    multi_true, multi_pred = [],[]
    for a in doc_couples_all:
        b.append(int(a/10000))
        
    for a in b:
        if b.count(a)==1:
            single.append(a)
    
    for pair in doc_couples_all:
        if int(pair/10000) not in single:
            multi_true.append(pair)
            doc_couples_all.remove(pair)
            
    for pair in doc_couples_pred_all:
        if int(pair/10000) not in single:
            multi_pred.append(pair)
            doc_couples_pred_all.remove(pair)
            
    p_single,r_single,f1_single = prf_2nd_step(doc_couples_all,doc_couples_pred_all)

    p,r,f1 = prf_2nd_step(multi_true,multi_pred)

    return p_single,r_single,f1_single,p,r,f1


def acc_prf(pred_y, true_y, mask,mask1, average='binary'):  # doc_len_b ,y_causes_b
    #print(pred_y)
    #print(pred_y.shape,true_y.shape)
    c = []
    for i,j in enumerate(mask1):
        if j==0:
            c.append(mask[i])
    mask1 = (1-torch.from_numpy(mask1).bool().long()).unsqueeze(1).bool()
    true_y = torch.masked_select(torch.Tensor(true_y),mask1).view(-1,true_y.shape[1])
    #print(true_y.shape,pred_y.shape)
    #print(max(pred_y))
    #print(max(true_y))
    tmp1, tmp2 = [], []
    for i in range(pred_y.shape[0]):
        for j in range(c[i]):
            tmp1.append(int(pred_y[i][j]))
            tmp2.append(int(true_y[i][j]))
    #print(tmp1,tmp2)
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1

def to_np(x):
    return x.data.cpu().numpy()


def logistic(x):
    return 1 / (1 + math.exp(-x))

def prf_2nd_step(pair_id_all, pred_y):
    s1, s3 = set(pair_id_all), set(pred_y)
    # print(s3)
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def eval_func(doc_couples_all, doc_couples_pred_all):
    tmp_num = {'ec': 0, 'e': 0, 'c': 0}
    tmp_den_p = {'ec': 0, 'e': 0, 'c': 0}
    tmp_den_r = {'ec': 0, 'e': 0, 'c': 0}

    doc_couples = set(doc_couples_all)
    doc_couples_pred = set(doc_couples_pred_all)

    tmp_num['ec'] += len(doc_couples & doc_couples_pred)
    tmp_den_p['ec'] += len(doc_couples_pred)
    tmp_den_r['ec'] += len(doc_couples) 

    doc_emos = set([int(int(doc_couple)/100) for doc_couple in doc_couples])
    doc_emos_pred = set([int(int(doc_couple)/100) for doc_couple in doc_couples_pred])
    tmp_num['e'] += len(doc_emos & doc_emos_pred)
    tmp_den_p['e'] += len(doc_emos_pred)
    tmp_den_r['e'] += len(doc_emos)

    doc_caus = set([int(doc_couple)%10+int(doc_couple/10000)*10000 for doc_couple in doc_couples])
    doc_caus_pred = set([int(doc_couple)%10+int(doc_couple/10000)*10000 for doc_couple in doc_couples_pred])

    tmp_num['c'] += len(doc_caus & doc_caus_pred)
    tmp_den_p['c'] += len(doc_caus_pred)
    tmp_den_r['c'] += len(doc_caus)

    metrics = {}
    for task in ['ec', 'e', 'c']:
        p = tmp_num[task] / (tmp_den_p[task] + 1e-8)
        r = tmp_num[task] / (tmp_den_r[task] + 1e-8)
        f = 2 * p * r / (p + r + 1e-8)
        metrics[task] = (p, r, f)

    return metrics['e'], metrics['c']

def float_n(value, n='0.0000'):
    value = decimal.Decimal(str(value)).quantize(decimal.Decimal(n))
    return float(value)


def write_b(b, b_path):
    with open(b_path, 'wb') as fw:
        pickle.dump(b, fw)


def read_b(b_path):
    with open(b_path, 'rb') as fr:
        b = pickle.load(fr)
    return b


def read_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as fr:
        js = json.load(fr)
    return js
