import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_SEED = 129
DATA_DIR = 'data'
TRAIN_FILE = 'fold%s_train.json'
VALID_FILE = 'fold%s_valid.json'
TEST_FILE  = 'fold%s_test.json'

SENTIMENTAL_CLAUSE_DICT = 'sentimental_clauses.pkl'


class Config(object):
    def __init__(self):
        self.split = 'split10'

        self.bert_cache_path = 'src/bert-base-chinese'
        self.feat_dim = 768

        self.epochs = 20

        self.batch_size = 8
        self.gradient_accumulation_steps = 1
        self.warmup_proportion = 0.1
        self.l2_bert = 1e-2
        self.l2_other = 1e-5
        self.lr = 2e-5

