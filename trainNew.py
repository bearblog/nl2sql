import os
import random
import copy
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import time
import math
import gc
import re
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
import json
from sklearn.metrics import *

class TrainerNL2SQL:
    def __init__(self, data_dir, epochs=1, batch_size=64, base_batch_size=32, max_seq_len=120 , seed=1234, debug = False):
        self.device = torch.device('cuda')
        self.data_dir = data_dir
        self.debug = debug
        self.seed = seed
        self.seed_everything()
        self.max_seq_len = max_seq_len
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        if not os.path.exists(self.data_dir):
            raise NotImplementedError()
        else :
            self.train_data_path = os.path.join(self.data_dir, "train/train.json")
            self.train_table_path = os.path.join(self.data_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(self.data_dir, "val/val.json")
            self.valid_table_path = os.path.join(self.data_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(self.data_dir, "test/test.json")
            self.test_table_path = os.path.join(self.data_dir, "test/test.tables.json")
            self.bert_model_path = "./chinese_wwm_ext_pytorch/"
            self.pytorch_bert_path =  "./chinese_wwm_ext_pytorch/pytorch_model.bin"
            self.bert_config = BertConfig("./chinese_wwm_ext_pytorch/bert_config.json")

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        # 固定随机数的种子保证结果可复现性
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def read_query(self,query_path):
        '''
        query_path 是带有用户问题的json 文件路径
        '''
        data = []
        with open (query_path,'r') as data_file:
            for line_index , each_line in enumerate(data_file):
                # debug 只读100行即可
                if self.debug and line_index == 100: break
                data.append(json.loads(each_line)) 
            print(len(data))
        return data
    def read_table(self,table_path):
        '''
        table_path 是对应于问题的存有完整数据库的json文件
        '''
        table = {}
        with open(table_path,'r') as table_file:
            for line_index,each_line in enumerate(table_file):
                table = json.loads(each_line)
                table[table['id']] = table
        return table



    

if __name__ == "__main__":
    data_dir = "./data"
    trainer = Trainer(data_dir, epochs=15, batch_size=16, base_batch_size=16, max_seq_len=128, debug = False)
    time1 = time.time()
    trainer.train()
    print("训练时间: %d min" % int((time.time() - time1) / 60))