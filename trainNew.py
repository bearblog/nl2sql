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

from utils import QuestionMatcher

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
            lines = data_file.readlines()
            if self.debug:
                data = [json.load(line) for line in lines if len(data) <= 100]
            else:
                data = [json.load(line) for line in lines]
            # for line_index , each_line in enumerate(data_file):
            #     # debug 只读100行即可
            #     if self.debug and line_index == 100: break
            #     data.append(json.loads(each_line))
            # print(len(data))
        return data

    def read_table(self,table_path):
        '''
        table_path 是对应于问题的存有完整数据库的json文件
        '''
        table = {}
        with open(table_path,'r') as table_file:
            lines = table_file.readlines()
            tables = [json.load(each_table) for each_table in lines]
            for each_table in tables:
                table[each_table['id']] = each_table
            # for line_index,each_line in enumerate(table_file):
            #     each_table = json.loads(each_line)
            #     table[each_table['id']] = each_table
        return table

    def create_mask(self,max_len,start_index,mask_len):
        '''
        对给定的序列中返回他对应的 mask 序列
        只保留起始索引到mask 长度的序列为1 ，其余为0
        '''
        mask = [0] * max_len
        for  index in range(start_index,start_index + mask_len):
            mask[index] = 1
        return mask

    def data_process(self,query,table,bert_tokenizer):
        question = query['question']
        tableID = query['table_id']
        sel_col = query['sql']['sel']
        sel_agg = query['sql']['agg']
        where_conds = query['sql']['conds']
        where_relation = query['sql']['cond_conn_op']
        '''
        table[tableID]
        {'rows': [['死侍2：我爱我家', 10637.3, 25.8, 5.0], ['白蛇：缘起', 10503.8, 25.4, 7.0], ['大黄蜂', 6426.6, 15.6, 6.0], ['密室逃生', 5841.4, 14.2, 6.0], ['“大”人物', 3322.9, 8.1, 5.0], ['家和万事惊', 635.2, 1.5, 25.0], ['钢铁飞龙之奥特曼崛起', 595.5, 1.4, 3.0], ['海王', 500.3, 1.2, 5.0], ['一条狗的回家路', 360.0, 0.9, 4.0], ['掠食城市', 356.6, 0.9, 3.0]], 'name': 'Table_4d29d0513aaa11e9b911f40f24344a08', 'title': '表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10', 'header': ['影片名称', '周票房（万）', '票房占比（%）', '场均人次'], 
        'common': '资料来源：艺恩电影智库，光大证券研究所', 'id': '4d29d0513aaa11e9b911f40f24344a08', 'types': ['text', 'real', 'real', 'real']}
        '''
        header_list = table[tableID]['header']
        row_list = table[tableID]["rows"]
        '''
        row_list: 是数据库中具体存放的行列
        [['死侍2：我爱我家', 10637.3, 25.8, 5.0], ['白蛇：缘起', 10503.8, 25.4, 7.0], ['大黄蜂', 6426.6, 15.6, 6.0], ['密室逃生', 5841.4, 14.2, 6.0], ['“大”人物', 3322.9, 8.1, 5.0], ['家和万事惊', 635.2, 1.5, 25.0], ['钢铁飞龙之奥特曼崛起', 595.5, 1.4, 3.0], ['海王', 500.3, 1.2, 5.0], ['一条狗的回家路', 360.0, 0.9, 4.0], ['掠食城市', 356.6, 0.9, 3.0]]
        '''
        col_dict = {header_name: set() for header_name in header_list}
        for row in row_list:
            for col, value in enumerate(row):
                header_name = header_list[col]
                col_dict[header_name].add(str(value))
        '''
        col_dict: 将数据库中列-value 整理成字典的形式
        {'指标': {'自筹资金', '房地产开发企业本年资金来源（亿元）', '个人按揭贷款', '定金及预收款', '国内贷款', '其他资金', '利用外资'}, 
        '绝对量': {'3343.0', '168.0', '13188.0', '14518.0', '34171.0', '7926.0', '6296.0'}, 
        '同比增长（%）': {'-36.8', '-4.0', '-2.9', '5.7', '8.5', '16.3', '-4.3'}}
        '''

        sel_num = len(sel_col)
        where_num = len(where_conds)
        # sel_dict -> {2: 5}
        sel_dict = {sel: agg for sel, agg in zip(sel_col, sel_agg)}

        duplicate_indices = QuestionMatcher.duplicate_relative_index(where_conds)






        return 1
    def data_iterator(self):
        train_data_list = self.read_query(self.train_data_path)
        train_table_dict = self.read_table(self.train_table_path)
        valid_data_list = self.read_query(self.valid_data_path)
        valid_table_dict = self.read_table(self.valid_table_path)
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
        for each_trainData in train_data_list:
            processed_result = self.data_process(each_trainData, train_table_dict, bert_tokenizer)



    

if __name__ == "__main__":
    data_dir = "./data"
    trainer = TrainerNL2SQL(data_dir, epochs=15, batch_size=16, base_batch_size=16, max_seq_len=128, debug = True)
    time1 = time.time()
    trainer.data_iterator()
    print("训练时间: %d min" % int((time.time() - time1) / 60))