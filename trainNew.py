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

'''
class InputFeaturesLabels:
    def __init__(
        self,connect_inputIDs,sequence_labeling_inputMask,sel_column_mask,where_conlumn_inputMask,type_mask,attention_mask,firstColumn_CLS_startPosition,
        question,table_id,header_mask,question_mask,nextColumn_CLS_startPosition,nextColumn_inputMask,value_mask,
        where_relation_label,sel_agg_label,sequence_labeling_label,where_conlumn_number_label,op_label,type_label,sel_num_label,where_num_label):

        # Features
        self.connect_inputIDs = connect_inputIDs
        self.sequence_labeling_inputMask = sequence_labeling_inputMask
        self.sel_column_mask = sel_column_mask
        self.where_conlumn_inputMask = where_conlumn_inputMask
        self.type_mask = type_mask
        self.attention_mask = attention_mask
        self.firstColumn_CLS_startPosition = firstColumn_CLS_startPosition
        self.question = question
        self.table_id  = table_id
        self.header_mask = header_mask
        self.question_mask = question_mask
        self.nextColumn_CLS_startPosition = nextColumn_CLS_startPosition
        self.nextColumn_inputMask = nextColumn_inputMask
        self.value_mask = value_mask

        # Labels   
        self.where_relation_label = where_relation_label
        self.sel_agg_label = sel_agg_label
        self.sequence_labeling_label = sequence_labeling_label
        self.where_conlumn_number_label = where_conlumn_number_label
        self.op_label = op_label
        self.type_label = type_label
        self.sel_num_label = sel_num_label
        self.where_num_label = where_num_label
'''

class InputFeaturesLabels:
    def __init__(self):

        # Features
        self.connect_inputIDs = []
        self.sequence_labeling_inputMask = []
        self.sel_column_mask = []
        self.where_conlumn_inputMask = []
        self.type_mask = []
        self.attention_mask = []
        self.firstColumn_CLS_startPosition = []
        self.question = []
        self.table_id  = []
        self.header_mask = []
        self.question_mask = []
        self.nextColumn_CLS_startPosition = []
        self.nextColumn_inputMask = []
        self.value_mask = []
        # Labels   
        self.where_relation_label = []
        self.sel_agg_label = []
        self.sequence_labeling_label = []
        self.where_conlumn_number_label = []
        self.op_label = []
        self.type_label = []
        self.sel_num_label = []
        self.where_num_label = []
class InputFeaturesLabelsForTrain(InputFeaturesLabels):
    def __init__(self):
        self.each_trainData_index = []
        self.sql_label = []
        self.quesion_list  = []
        self.table_id_list = []


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
            # lines = table_file.readlines()
            # tables = [json.load(each_table) for each_table in lines]
            # for each_table in tables:
            #     table[each_table['id']] = each_table
            for line_index,each_line in enumerate(table_file):
                each_table = json.loads(each_line)
                table[each_table['id']] = each_table
        return table

    def create_mask(self,max_seq_len,start_index,mask_len):
        '''
        对给定的序列中返回他对应的 mask 序列
        只保留起始索引到mask 长度的序列为1 ，其余为0
        '''
        mask = [0] * max_seq_len
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

        sel_num_label = len(sel_col)
        where_num_label = len(where_conds)
        # sel_dict -> {2: 5}
        sel_dict = {sel: agg for sel, agg in zip(sel_col, sel_agg)}
        duplicate_indices = QuestionMatcher.duplicate_relative_index(where_conds)

        condition_dict = {}
        for [where_col,where_op,where_value] , duplicate_index in zip(where_conds,duplicate_indices):
            where_value = where_value.strip()
            matched_value,matched_index  = QuestionMatcher.match_value(question,where_value,duplicate_index)
            '''
            question                二零一九年第四周大黄蜂和密室逃生这两部影片的票房总占比是多少呀
            matched_value           大黄蜂
            match_index             8
            '''
            if len(matched_value) >0:
                if where_col in condition_dict:
                    condition_dict[where_col].append([where_op, matched_value, matched_index])
                else:
                    condition_dict[where_col] = [[where_op, matched_value, matched_index]]
            else:
                # TODO  是否存在匹配不到值的情况，以及该如何处理
                pass
            # condition_dict : {0: [[2, '大黄蜂', 8]]}

        features_labels = InputFeaturesLabels()
        
        question_ = bert_tokenizer.tokenize(question)
        question_inputIDs = bert_tokenizer.convert_tokens_to_ids (['[CLS]']+question_+['[SEP]'])
        firstColumn_CLS_startPosition = len(question_inputIDs)
        question_inputMask = self.create_mask(max_seq_len = self.max_seq_len,start_index = 1,mask_len = len(question_))
        '''
        question_mask 就是对于输入的序列将对应的问题部分打上mask标记
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        '''
        for index_header in range(len(header_list)):
            # each_column : 影片名称
            each_column = header_list[index_header]
            value_dict = col_dict[each_column]
            print(each_column)
            '''
            each_column = 影片名称
            value_dict = {'家和万事惊', '密室逃生', '钢铁飞龙之奥特曼崛起', '海王', '“大”人物', '掠食城市', '死侍2：我爱我家', '大黄蜂', '一条狗的回家路', '白蛇：缘起'}
            '''
            each_column_ = bert_tokenizer.tokenize(each_column)
            each_column_inputIDs = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + each_column_ + ["[SEP]"])
            each_column_inputMask =self.create_mask(max_seq_len = self.max_seq_len,start_index = len(question_inputIDs)+1,mask_len  = len(each_column_))

            connect_inputIDs = question_inputIDs + each_column_inputIDs

            # question + column 后面再接的column对应的CLS的索引
            nextColumn_CLS_startPosition = len(connect_inputIDs)
            # 后面的column的 起始索引
            nextColumn_startPosition = nextColumn_CLS_startPosition + 1
            random.seed(index_header)

            for index_nextColumn,nextColumn in enumerate(random.sample(header_list,len(header_list))):
                nextColumn_ = bert_tokenizer.tokenize(nextColumn)
                if index_nextColumn == 0:
                    nextColumn_inputIDs = bert_tokenizer.convert_tokens_to_ids(['[CLS]']+nextColumn_+['[SEP]'])
                else:
                    nextColumn_inputIDs = bert_tokenizer.convert_tokens_to_ids(nextColumn_+['[SEP]'])
                if len(connect_inputIDs) + len(nextColumn_inputIDs) <= self.max_seq_len:
                    connect_inputIDs +=nextColumn_inputIDs
                else:
                    break

            # nextColumn_inputMask_len 要mask掉的后面的列的长度
            nextColumn_inputMask_len = len(connect_inputIDs) - nextColumn_startPosition - 1
            nextColumn_inputMask = self.create_mask(max_seq_len= self.max_seq_len,start_index =nextColumn_startPosition,mask_len = nextColumn_inputMask_len)
            '''
            nextColumn_inputMask 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            '''

            value_CLS_startPosition = len(connect_inputIDs)
            value_startPosition = len(connect_inputIDs) + 1
            print(value_dict)
            for value_index,each_value in enumerate(value_dict):
                each_value_ = bert_tokenizer.tokenize(each_value)
                if value_index == 0:
                    value_inputIDs = bert_tokenizer.convert_tokens_to_ids(['[CLS]']+each_value_+['[SEP]'])
                else:
                    value_inputIDs = bert_tokenizer.convert_tokens_to_ids(each_value_+['[SEP]'])
                if len(connect_inputIDs) + len(value_inputIDs) <= self.max_seq_len:
                    connect_inputIDs += value_inputIDs
                else:
                    break

            value_inputMask_len = len(connect_inputIDs) - value_startPosition -1 
            value_inputMask = self.create_mask(max_seq_len = self.max_seq_len,start_index = value_startPosition,mask_len = value_inputMask_len)
            
            # 此时connect_inputIDs 相当于 CLS + query+ SEP +column1+ SEP+ nextcolumn1 +SEP + nextcolumn2+ SEP + value1 + SEP
            # value 对应的是数据库里当前header 下面的全部的value
            # attention_mask 是 对当前是connet_inputIDs 做mask
            attention_mask = self.create_mask(max_seq_len = self.max_seq_len,start_index = 0,mask_len = len(connect_inputIDs))
            # padding
            connect_inputIDs = connect_inputIDs + [0]*(self.max_seq_len - len(connect_inputIDs))

            # 初始化序列标注的 input_id
            sequence_labeling_label = [0]*len(connect_inputIDs) 

            sel_column_mask, where_conlumn_inputMask, type_mask = 0, 0, 1
            # TODO op_label 一直都是2是不是有问题？
            where_relation_label, sel_agg_label, where_conlumn_number_label, op_label = 0, 0, 0, 2
            '''
            condition_dict 
            {0: [[2, '大黄蜂', 8], [2, '密室逃生', 12]]}
            '''
            if index_header in condition_dict:
                # 对于一个header 多个value的情况，必须是value的数量对应上才进入训练，否则continue
                # TODO 这地方是不是可以优化一下
                if list(map(lambda x:x[0],where_conds)).count(index_header) != len(condition_dict[index_header]): continue
                conlumn_condition_list = condition_dict[index_header]

                for [conlumn_condition_op,conlumn_condition_value,conlumn_condition_index] in conlumn_condition_list:

                    # 序列标注将问题question中value对应的部分标注成1
                    sequence_labeling_label[conlumn_condition_index+1:conlumn_condition_index+1+len(conlumn_condition_value)] = [1] * len(conlumn_condition_value)
                    # TODO 序列标注inputID 是问题中的value ,inpustMask是整个问题？ 
                sequence_labeling_inputMask = [0] + [1]*len(question) +[0]*(self.max_seq_len-len(question)-1)
                where_conlumn_inputMask = 1
                where_relation_label = where_relation
                where_conlumn_number_label = min(len(conlumn_condition_list),3) # 一个列对应4个value的只有一个样本，把他剔除掉
                type_label = 1

            elif index_header in sel_dict:
                sequence_labeling_inputMask = [0]*self.max_seq_len
                sel_column_mask = 1
                sel_agg_label = sel_dict[index_header]
                type_label = 0
            else:
                sequence_labeling_inputMask = [0]*self.max_seq_len
                type_label = 2
            '''
            这里相当于挨个遍历header_list中的列然后依次给对应的变量打上标签
            如果当前的列在condition_dict 也就是在conds 中，那么给对应的问题打上序列标注的标签
            type_label 是用来空值当前的列对应的是 sel的列还是 where 里的列，如果标记为2 那么就表示不选择当前的这个列
            '''

            
            # features
            features_labels.connect_inputIDs.append(connect_inputIDs)
            features_labels.sequence_labeling_inputMask.append(sequence_labeling_inputMask)
            features_labels.sel_column_mask.append(sel_column_mask)
            features_labels.where_conlumn_inputMask.append(where_conlumn_inputMask)
            features_labels.type_mask.append(type_mask)
            features_labels.attention_mask.append(attention_mask)
            features_labels.firstColumn_CLS_startPosition.append(firstColumn_CLS_startPosition)
            features_labels.question.append(question)
            features_labels.table_id.append(tableID)
            features_labels.header_mask.append(each_column_inputMask)
            features_labels.question_mask.append(question_inputMask)
            features_labels.nextColumn_CLS_startPosition.append(nextColumn_CLS_startPosition)
            features_labels.nextColumn_inputMask.append(nextColumn_inputMask)
            features_labels.value_mask.append(value_inputMask)
            # labels
            features_labels.where_relation_label.append(where_relation_label)
            features_labels.sel_agg_label.append(sel_agg_label)
            features_labels.sequence_labeling_label.append(sequence_labeling_label)
            # 剔除掉 一个列对应多个value 但是标注不一致 以后剩下的
            features_labels.where_conlumn_number_label.append(where_conlumn_number_label)
            features_labels.op_label.append(op_label)
            features_labels.type_label.append(type_label)
            features_labels.sel_num_label.append(sel_num_label)
            # 原始的where_condition 中 num 的数量
            features_labels.where_num_label.append(where_num_label)

        return features_labels


    def data_iterator(self):
        train_data_list = self.read_query(self.train_data_path)
        train_table_dict = self.read_table(self.train_table_path)
        valid_data_list = self.read_query(self.valid_data_path)
        valid_table_dict = self.read_table(self.valid_table_path)
        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

        train_features_labels = InputFeaturesLabelsForTrain()

        for each_trainData in train_data_list:
            features_labels = self.data_process(each_trainData, train_table_dict, bert_tokenizer)
            train_features_labels.connect_inputIDs.extend(features_labels.connect_inputIDs)
            train_features_labels.sequence_labeling_inputMask.extend(features_labels.sequence_labeling_inputMask)
            train_features_labels.sel_column_mask.extend(features_labels.sel_column_mask)
            train_features_labels.where_conlumn_inputMask.extend(features_labels.where_conlumn_inputMask)
            train_features_labels.type_mask.extend(features_labels.type_mask)
            train_features_labels.attention_mask.extend(features_labels.attention_mask)
            train_features_labels.firstColumn_CLS_startPosition.extend(features_labels.firstColumn_CLS_startPosition)
            train_features_labels


            train_features_labels


            train_features_labels


            train_features_labels



            train_features_labels

            train_features_labels

            train_features_labels

            train_features_labels


            





    

if __name__ == "__main__":
    data_dir = "./data"
    trainer = TrainerNL2SQL(data_dir, epochs=15, batch_size=16, base_batch_size=16, max_seq_len=128, debug = False)
    time1 = time.time()
    trainer.data_iterator()
    print("训练时间: %d min" % int((time.time() - time1) / 60))