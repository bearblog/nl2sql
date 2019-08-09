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

from utils import QuestionMatcher,MatrixAttentionLayer,ColAttentionLayer,ValueOptimizer


class BertNeuralNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNeuralNet, self).__init__(config)
        self.num_tag_labels = 2
        self.num_agg_labels = 6
        self.num_connection_labels = 3
        self.num_con_num_labels = 4
        self.num_type_labels = 3
        self.num_sel_num_labels = 4  # {1, 2, 3}
        self.num_where_num_labels = 5  # {1, 2, 3, 4}
        self.num_op_labels = 4

        op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}
        agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
        conn_sql_dict = {0: "", 1: "and", 2: "or"}
        con_num_dict = {0: 0, 1: 1, 2: 2, 3: 3}
        type_dict = {0: "sel", 1: "con", 2: "none"}
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear_tag = nn.Linear(self.hidden_size * 3, self.num_tag_labels)
        self.linear_agg = nn.Linear(self.hidden_size * 2, self.num_agg_labels)
        self.linear_connection = nn.Linear(self.hidden_size, self.num_connection_labels)
        self.linear_con_num = nn.Linear(self.hidden_size * 2, self.num_con_num_labels)
        self.linear_type = nn.Linear(self.hidden_size * 2, self.num_type_labels)
        self.linear_sel_num = nn.Linear(self.hidden_size, self.num_sel_num_labels)
        self.linear_where_num = nn.Linear(self.hidden_size, self.num_where_num_labels)
        self.values_attention = MatrixAttentionLayer(self.hidden_size, self.hidden_size)
        self.head_attention = ColAttentionLayer(self.hidden_size, self.hidden_size)
        self.linear_op = nn.Linear(self.hidden_size * 2, self.num_op_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask, all_masks, header_masks, question_masks, subheader_masks, subheader_cls_list, value_masks, cls_index_list, train_dependencies=None):
        sequence_output, _ = self.bert(input_ids, None, attention_mask, output_all_encoded_layers=False)

        device = "cuda" if torch.cuda.is_available() else None
        type_masks = all_masks.view(-1) == 1
        # TODO: 如果是求平均值就计算每行mask总和，mask，每行相加除以每行总和
        cls_output = sequence_output[type_masks, cls_index_list[type_masks], :]
        _, subheader_attention = self.head_attention(cls_output, sequence_output, subheader_masks)
        cat_cls = torch.cat([cls_output, subheader_attention], 1)

        cat_output, _ = self.values_attention(sequence_output, question_masks, sequence_output, value_masks)
        _, header_attention = self.values_attention(sequence_output, question_masks, sequence_output, header_masks)
        cat_output = torch.cat([cat_output, header_attention], 2)

        num_output = sequence_output[type_masks, 0, :]
        if train_dependencies:
            tag_masks = train_dependencies[0].view(-1) == 1      # 必须要加 view 和 == 1
            sel_masks = train_dependencies[1].view(-1) == 1
            con_masks = train_dependencies[2].view(-1) == 1
            type_masks = all_masks.view(-1) == 1
            connection_labels = train_dependencies[3]
            agg_labels = train_dependencies[4]
            tag_labels = train_dependencies[5]
            con_num_labels = train_dependencies[6]
            type_labels = train_dependencies[7]
            sel_num_labels = train_dependencies[8]
            where_num_labels = train_dependencies[9]
            op_labels = train_dependencies[10]
            # mask 后的 bert_output
            tag_output = cat_output.contiguous().view(-1, self.hidden_size * 3)[tag_masks]
            tag_labels = tag_labels.view(-1)[tag_masks]
            agg_output = cat_cls[sel_masks, :]
            agg_labels = agg_labels[sel_masks]
            connection_output = sequence_output[con_masks, 0, :]
            connection_labels = connection_labels[con_masks]
            con_num_output = cat_cls[con_masks, :]
            con_num_labels = con_num_labels[con_masks]
            op_output = cat_cls[con_masks, :]
            op_labels = op_labels[con_masks]
            type_output = cat_cls[type_masks, :]
            type_labels = type_labels[type_masks]
            # 全连接层
            tag_output = self.linear_tag(self.dropout(tag_output))
            agg_output = self.linear_agg(self.dropout(agg_output))
            connection_output = self.linear_connection(self.dropout(connection_output))
            con_num_output = self.linear_con_num(self.dropout(con_num_output))
            type_output = self.linear_type(self.dropout(type_output))
            sel_num_output = self.linear_sel_num(self.dropout(num_output))
            where_num_output = self.linear_where_num(self.dropout(num_output))
            op_output = self.linear_op(self.dropout(op_output))
            # 损失函数
            loss_function = nn.CrossEntropyLoss(reduction="mean")
            tag_loss = loss_function(tag_output, tag_labels)
            agg_loss = loss_function(agg_output, agg_labels)
            connection_loss = loss_function(connection_output, connection_labels)
            con_num_loss = loss_function(con_num_output, con_num_labels)
            type_loss = loss_function(type_output, type_labels)
            sel_num_loss = loss_function(sel_num_output, sel_num_labels)
            where_num_loss = loss_function(where_num_output, where_num_labels)
            op_loss = loss_function(op_output, op_labels)
            loss = tag_loss + agg_loss + connection_loss + con_num_loss + type_loss + sel_num_loss + where_num_loss + op_loss
            return loss
        else:
            all_masks = all_masks.view(-1) == 1
            batch_size, seq_len, hidden_size = sequence_output.shape
            tag_output = torch.zeros(batch_size, seq_len, hidden_size * 3, dtype=torch.float32, device=device)
            for i in range(batch_size):
                for j in range(seq_len):
                    if attention_mask[i][j] == 1:
                        tag_output[i][j] = cat_output[i][j]
            head_output = sequence_output[:, 0, :]
            # cls_output = sequence_output[all_masks, cls_index_list, :]
            tag_output = self.linear_tag(self.dropout(tag_output))
            agg_output = self.linear_agg(self.dropout(cat_cls))
            connection_output = self.linear_connection(self.dropout(head_output))
            con_num_output = self.linear_con_num(self.dropout(cat_cls))
            type_output = self.linear_type(self.dropout(cat_cls))
            sel_num_output = self.linear_sel_num(self.dropout(num_output))
            where_num_output = self.linear_where_num(self.dropout(num_output))
            op_output = self.linear_op(self.dropout(cat_cls))

            tag_probs = F.log_softmax(tag_output, dim=2).detach().cpu().numpy().tolist()
            agg_probs = F.log_softmax(agg_output, dim=1).detach().cpu().numpy().tolist()
            connection_probs = F.log_softmax(connection_output, dim=1).detach().cpu().numpy().tolist()
            con_num_probs = F.log_softmax(con_num_output, dim=1).detach().cpu().numpy().tolist()
            type_probs = F.log_softmax(type_output, dim=1).detach().cpu().numpy().tolist()
            sel_num_probs = F.log_softmax(sel_num_output, dim=1).detach().cpu().numpy().tolist()
            where_num_probs = F.log_softmax(where_num_output, dim=1).detach().cpu().numpy().tolist()
            op_probs = F.log_softmax(op_output, dim=1).detach().cpu().numpy().tolist()
            probs_list = [tag_probs, agg_probs, connection_probs, con_num_probs, type_probs, sel_num_probs, where_num_probs, op_probs]

            tag_logits = torch.argmax(F.log_softmax(tag_output, dim=2), dim=2).detach().cpu().numpy().tolist()
            agg_logits = torch.argmax(F.log_softmax(agg_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            connection_logits = torch.argmax(F.log_softmax(connection_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            con_num_logits = torch.argmax(F.log_softmax(con_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            type_logits = torch.argmax(F.log_softmax(type_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            sel_num_logits = torch.argmax(F.log_softmax(sel_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            where_num_logits = torch.argmax(F.log_softmax(where_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            op_logits = torch.argmax(F.log_softmax(op_output, dim=1), dim=1).detach().cpu().numpy().tolist()

            return tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits, probs_list

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
        self.question_tokens = []
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
        super(InputFeaturesLabelsForTrain, self).__init__()
        self.each_trainData_index = []
        self.sql_label = []
        self.quesion_list  = []
        self.table_id_list = []
class NL2SQL:
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
            # print(len(data))
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

    def data_process(self,query,table,bert_tokenizer,test = False):
        if test == False:
            question = query['question']
            tableID = query['table_id']
            select_column = query['sql']['sel']
            select_agg = query['sql']['agg']
            where_conditions = query['sql']['conds']
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
            columnValue_dict = {header_name: set() for header_name in header_list}
            for row in row_list:
                for column, value in enumerate(row):
                    header_name = header_list[column]
                    columnValue_dict[header_name].add(str(value))
            '''
            columnValue_dict: 将数据库中列-value 整理成字典的形式
            {'指标': {'自筹资金', '房地产开发企业本年资金来源（亿元）', '个人按揭贷款', '定金及预收款', '国内贷款', '其他资金', '利用外资'}, 
            '绝对量': {'3343.0', '168.0', '13188.0', '14518.0', '34171.0', '7926.0', '6296.0'}, 
            '同比增长（%）': {'-36.8', '-4.0', '-2.9', '5.7', '8.5', '16.3', '-4.3'}}
            '''

            select_number_label = len(select_column)
            where_number_label = len(where_conditions)
            # sel_clause_dict -> {2: 5}
            select_clause_dict = {column: agg_function for column, agg_function in zip(select_column, select_agg)}
            duplicate_indices = QuestionMatcher.duplicate_relative_index(where_conditions)

            condition_dict = {}
            for [where_col,where_op,where_value] , duplicate_index in zip(where_conditions,duplicate_indices):
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
            question_UNK_position = []
            question_ = bert_tokenizer.tokenize(question.strip().replace(' ',''))
            for index,each_token in enumerate(question_):
                if each_token == "[UNK]":
                    # TODO 似乎不存在 [UNK]
                    question_UNK_position.extend([index])
                    # print(question)
                    # exit()
                else:
                    question_UNK_position.extend([index]*len(each_token))
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
                value_dict = columnValue_dict[each_column]
                # print(each_column)
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
                # 这里的起始位置指的是connect_id中的索引
                value_CLS_startPosition = len(connect_inputIDs)
                value_startPosition = len(connect_inputIDs) + 1
                # print(value_dict)
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

                # 初始化序列标注的 ground truth
                sequence_labeling_label = [0]*len(connect_inputIDs) 

                select_column_mask, where_conlumn_inputMask, type_mask = 0, 0, 1
                # TODO op_label 一直都是2是不是有问题？
                where_relation_label, select_agg_label, where_conlumn_number_label, op_label = 0, 0, 0, 2
                '''
                condition_dict 
                {0: [[2, '大黄蜂', 8], [2, '密室逃生', 12]]}
                '''
                if index_header in condition_dict:
                    # 对于一个header 多个value的情况，必须是value的数量对应上才进入训练，否则continue
                    # TODO 这地方是不是可以优化一下,感觉就算是没有对上也可以进入训练吧
                    if list(map(lambda x:x[0],where_conditions)).count(index_header) != len(condition_dict[index_header]): continue
                    conlumn_condition_list = condition_dict[index_header]

                    for [conlumn_condition_op,conlumn_condition_value,conlumn_condition_index] in conlumn_condition_list:
                        value_startposition_inQuestion = conlumn_condition_index
                        # end_position : 8+len('大黄蜂') -1 = 10
                        value_endposition_inQuestion = conlumn_condition_index + len(conlumn_condition_value) -1
                        # 处理了一下UNK
                        value_startposition_forLabeling = question_UNK_position[value_startposition_inQuestion] + 1     # cls
                        value_endposition_forLabeling = question_UNK_position[value_endposition_inQuestion] +1 +1       # cls sep
                        # 序列标注将问题question中value对应的部分标注成1
                        sequence_labeling_label[value_startposition_forLabeling:value_endposition_forLabeling] = [1] * (value_endposition_forLabeling - value_startposition_forLabeling)
                        # TODO 序列标注inputID 是问题中的value ,inpustMask是整个问题？ 
                    sequence_labeling_inputMask = [0] + [1]*len(question_) +[0]*(self.max_seq_len-len(question_)-1)
                    where_conlumn_inputMask = 1
                    where_relation_label = where_relation
                    where_conlumn_number_label = min(len(conlumn_condition_list),3) # 一个列对应4个value的只有一个样本，把他剔除掉
                    type_label = 1

                elif index_header in select_clause_dict:
                    sequence_labeling_inputMask = [0]*self.max_seq_len
                    select_column_mask = 1
                    select_agg_label = select_clause_dict[index_header]
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
                features_labels.sel_column_mask.append(select_column_mask)
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
                features_labels.question_tokens.append(question_)
                # labels
                features_labels.where_relation_label.append(where_relation_label)
                features_labels.sel_agg_label.append(select_agg_label)
                features_labels.sequence_labeling_label.append(sequence_labeling_label)
                # 剔除掉 一个列对应多个value 但是标注不一致 以后剩下的
                features_labels.where_conlumn_number_label.append(where_conlumn_number_label)
                features_labels.op_label.append(op_label)
                features_labels.type_label.append(type_label)
                features_labels.sel_num_label.append(select_number_label)
                # 原始的where_condition 中 num 的数量
                features_labels.where_num_label.append(where_number_label)

        return features_labels


    def data_iterator(self):
        train_data = self.read_query(self.train_data_path)
        train_table = self.read_table(self.train_table_path)
        valid_data = self.read_query(self.valid_data_path)
        valid_table = self.read_table(self.valid_table_path)
        test_data = self.read_query(self.test_data_path)
        test_table = self.read_table(self.test_table_path)

        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

        train_features_labels = InputFeaturesLabelsForTrain()
        valid_features_labels = InputFeaturesLabelsForTrain()
        test_features_labels = InputFeaturesLabelsForTrain()


        for each_trainData in train_data:
            features_labels = self.data_process(each_trainData, train_table, bert_tokenizer)
            train_features_labels.connect_inputIDs.extend(features_labels.connect_inputIDs)
            train_features_labels.sequence_labeling_inputMask.extend(features_labels.sequence_labeling_inputMask)
            train_features_labels.sel_column_mask.extend(features_labels.sel_column_mask)
            train_features_labels.where_conlumn_inputMask.extend(features_labels.where_conlumn_inputMask)
            train_features_labels.type_mask.extend(features_labels.type_mask)
            train_features_labels.attention_mask.extend(features_labels.attention_mask)
            train_features_labels.firstColumn_CLS_startPosition.extend(features_labels.firstColumn_CLS_startPosition)
            train_features_labels.question.extend(features_labels.question)
            train_features_labels.table_id.extend(features_labels.table_id)
            train_features_labels.header_mask.extend(features_labels.header_mask)
            train_features_labels.question_mask.extend(features_labels.question_mask)
            train_features_labels.nextColumn_CLS_startPosition.extend(features_labels.nextColumn_CLS_startPosition)
            train_features_labels.nextColumn_inputMask.extend(features_labels.nextColumn_inputMask)
            train_features_labels.value_mask.extend(features_labels.value_mask)

            train_features_labels.where_relation_label.extend(features_labels.where_relation_label)
            train_features_labels.sel_agg_label.extend(features_labels.sel_agg_label)
            train_features_labels.sequence_labeling_label.extend(features_labels.sequence_labeling_label)
            train_features_labels.where_conlumn_number_label.extend(features_labels.where_conlumn_number_label)
            train_features_labels.op_label.extend(features_labels.op_label)
            train_features_labels.type_label.extend(features_labels.type_label)
            train_features_labels.sel_num_label.extend(features_labels.sel_num_label)
            train_features_labels.where_num_label.extend(features_labels.where_num_label)
            train_features_labels.question_tokens.append(features_labels.question_tokens)

            train_features_labels.each_trainData_index.append(len(train_features_labels.connect_inputIDs))
            train_features_labels.sql_label.append(each_trainData['sql'])
            train_features_labels.quesion_list.append(each_trainData['question'])
            train_features_labels.table_id_list.append(each_trainData['table_id'])

            
        for each_validData in valid_data:
            features_labels = self.data_process(each_validData, valid_table, bert_tokenizer)
            valid_features_labels.connect_inputIDs.extend(features_labels.connect_inputIDs)
            valid_features_labels.sequence_labeling_inputMask.extend(features_labels.sequence_labeling_inputMask)
            valid_features_labels.sel_column_mask.extend(features_labels.sel_column_mask)
            valid_features_labels.where_conlumn_inputMask.extend(features_labels.where_conlumn_inputMask)
            valid_features_labels.type_mask.extend(features_labels.type_mask)
            valid_features_labels.attention_mask.extend(features_labels.attention_mask)
            valid_features_labels.firstColumn_CLS_startPosition.extend(features_labels.firstColumn_CLS_startPosition)
            valid_features_labels.question.extend(features_labels.question)
            valid_features_labels.table_id.extend(features_labels.table_id)
            valid_features_labels.header_mask.extend(features_labels.header_mask)
            valid_features_labels.question_mask.extend(features_labels.question_mask)
            valid_features_labels.nextColumn_CLS_startPosition.extend(features_labels.nextColumn_CLS_startPosition)
            valid_features_labels.nextColumn_inputMask.extend(features_labels.nextColumn_inputMask)
            valid_features_labels.value_mask.extend(features_labels.value_mask)

            valid_features_labels.where_relation_label.extend(features_labels.where_relation_label)
            valid_features_labels.sel_agg_label.extend(features_labels.sel_agg_label)
            valid_features_labels.sequence_labeling_label.extend(features_labels.sequence_labeling_label)
            valid_features_labels.where_conlumn_number_label.extend(features_labels.where_conlumn_number_label)
            valid_features_labels.op_label.extend(features_labels.op_label)
            valid_features_labels.type_label.extend(features_labels.type_label)
            valid_features_labels.sel_num_label.extend(features_labels.sel_num_label)
            valid_features_labels.where_num_label.extend(features_labels.where_num_label)
            valid_features_labels.question_tokens.append(features_labels.question_tokens)


            valid_features_labels.each_trainData_index.append(len(valid_features_labels.connect_inputIDs))
            valid_features_labels.sql_label.append(each_trainData['sql'])
            valid_features_labels.quesion_list.append(each_trainData['question'])
            valid_features_labels.table_id_list.append(each_trainData['table_id'])

        '''
        for each_testData in test_data_list:
            features_labels = self.data_process(each_testData, test_table_dict, bert_tokenizer)
            test_features_labels.connect_inputIDs.extend(features_labels.connect_inputIDs)
            test_features_labels.type_mask.extend(features_labels.type_mask)
            test_features_labels.attention_mask.extend(features_labels.attention_mask)
            test_features_labels.firstColumn_CLS_startPosition.extend(features_labels.firstColumn_CLS_startPosition)
            test_features_labels.header_mask.extend(features_labels.header_mask)
            test_features_labels.question_mask.extend(features_labels.question_mask)
            test_features_labels.nextColumn_CLS_startPosition.extend(features_labels.nextColumn_CLS_startPosition)
            test_features_labels.nextColumn_inputMask.extend(features_labels.nextColumn_inputMask)
            test_features_labels.value_mask.extend(features_labels.value_mask)

            test_features_labels.each_trainData_index.append(len(train_features_labels.connect_inputIDs))
            test_features_labels.quesion_list.append(each_trainData['question'])
            test_features_labels.table_id_list.append(each_trainData['table_id'])
        '''

        train_data = data.TensorDataset(
            torch.tensor(train_features_labels.connect_inputIDs, dtype=torch.long),
            torch.tensor(train_features_labels.sequence_labeling_inputMask, dtype=torch.long),
            torch.tensor(train_features_labels.sel_column_mask, dtype=torch.long),
            torch.tensor(train_features_labels.where_conlumn_inputMask, dtype=torch.long),
            torch.tensor(train_features_labels.type_mask, dtype=torch.long),
            torch.tensor(train_features_labels.attention_mask, dtype=torch.long),
            torch.tensor(train_features_labels.where_relation_label, dtype=torch.long),
            torch.tensor(train_features_labels.sel_agg_label, dtype=torch.long),
            torch.tensor(train_features_labels.sequence_labeling_label, dtype=torch.long),
            torch.tensor(train_features_labels.where_conlumn_number_label, dtype=torch.long),
            torch.tensor(train_features_labels.type_label, dtype=torch.long),
            torch.tensor(train_features_labels.firstColumn_CLS_startPosition, dtype=torch.long),
            torch.tensor(train_features_labels.header_mask, dtype=torch.long),
            torch.tensor(train_features_labels.question_mask, dtype=torch.long),
            torch.tensor(train_features_labels.nextColumn_CLS_startPosition, dtype=torch.long),
            torch.tensor(train_features_labels.nextColumn_inputMask, dtype=torch.long),
            torch.tensor(train_features_labels.sel_num_label, dtype=torch.long),
            torch.tensor(train_features_labels.where_num_label, dtype=torch.long),
            torch.tensor(train_features_labels.op_label, dtype=torch.long),
            torch.tensor(train_features_labels.value_mask, dtype=torch.long)
            )

        valid_data = data.TensorDataset(
            torch.tensor(valid_features_labels.connect_inputIDs, dtype=torch.long),
            torch.tensor(valid_features_labels.sequence_labeling_inputMask, dtype=torch.long),
            torch.tensor(valid_features_labels.sel_column_mask, dtype=torch.long),
            torch.tensor(valid_features_labels.where_conlumn_inputMask, dtype=torch.long),
            torch.tensor(valid_features_labels.type_mask, dtype=torch.long),
            torch.tensor(valid_features_labels.attention_mask, dtype=torch.long),
            torch.tensor(valid_features_labels.where_relation_label, dtype=torch.long),
            torch.tensor(valid_features_labels.sel_agg_label, dtype=torch.long),
            torch.tensor(valid_features_labels.sequence_labeling_label, dtype=torch.long),
            torch.tensor(valid_features_labels.where_conlumn_number_label, dtype=torch.long),
            torch.tensor(valid_features_labels.type_label, dtype=torch.long),
            torch.tensor(valid_features_labels.firstColumn_CLS_startPosition, dtype=torch.long),
            torch.tensor(valid_features_labels.header_mask, dtype=torch.long),
            torch.tensor(valid_features_labels.question_mask, dtype=torch.long),
            torch.tensor(valid_features_labels.nextColumn_CLS_startPosition, dtype=torch.long),
            torch.tensor(valid_features_labels.nextColumn_inputMask, dtype=torch.long),
            torch.tensor(valid_features_labels.sel_num_label, dtype=torch.long),
            torch.tensor(valid_features_labels.where_num_label, dtype=torch.long),
            torch.tensor(valid_features_labels.op_label, dtype=torch.long),
            torch.tensor(valid_features_labels.value_mask, dtype=torch.long)
            )

        '''
        test_data = data.TensorDataset(
            torch.tensor(test_features_labels.connect_inputIDs, dtype=torch.long),
            torch.tensor(test_features_labels.attention_mask, dtype=torch.long),
            torch.tensor(test_features_labels.firstColumn_CLS_startPosition, dtype=torch.long),
            torch.tensor(test_features_labels.header_mask, dtype=torch.long),
            torch.tensor(test_features_labels.question_mask, dtype=torch.long),
            torch.tensor(test_features_labels.nextColumn_CLS_startPosition, dtype=torch.long),
            torch.tensor(test_features_labels.nextColumn_inputMask, dtype=torch.long),
            torch.tensor(test_features_labels.value_mask, dtype=torch.long),
            torch.tensor(test_features_labels.type_mask, dtype=torch.long),
                                          )
        '''
        # 迭代器迭代出每一个batch的数据
        train_iterator = torch.utils.data.DataLoader(train_data, batch_size=self.base_batch_size, shuffle=True)
        valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=self.base_batch_size, shuffle=False)
        # test_iterator = torch.utils.data.DataLoader(test_data, batch_size=self.base_batch_size, shuffle=False)

        return train_iterator, valid_iterator,valid_features_labels,valid_table
        # return train_loader, valid_loader, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list, test_loader, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict
    def evaluate(self, logits_lists, cls_index_list, labels_lists, question_list, question_token_list, table_id_list, sample_index_list, correct_sql_list, table_dict, header_question_list, header_table_id_list, probs_list, do_test=False):
        [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list, op_logits_list] = logits_lists
        [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list, type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list] = labels_lists
        [tag_probs_list, agg_probs_list, connection_probs_list, con_num_probs_list, type_probs_list, sel_num_probs_list, where_num_probs_list, op_probs_list] = probs_list

        f_valid = open("valid_detail.txt", 'w')
        # {"agg": [0], "cond_conn_op": 2, "sel": [1], "conds": [[3, 0, "11"], [6, 0, "11"]]}
        sql_dict = {"agg": [], "cond_conn_op": None, "sel": [], "conds": []}
        sql_list = []
        matched_num = 0
        for i in range(len(sample_index_list)):
            start_index = 0 if i == 0 else sample_index_list[i - 1]
            end_index = sample_index_list[i]
            sample_question = question_list[i]
            sample_question_token = question_token_list[i]
            sample_table_id = table_id_list[i]
            if do_test is False:
                sample_sql = correct_sql_list[i]
            sample_tag_logits = tag_logits_list[start_index: end_index]
            sample_agg_logits = agg_logits_list[start_index: end_index]
            sample_connection_logits = connection_logits_list[start_index: end_index]
            sample_con_num_logits = con_num_logits_list[start_index: end_index]
            sample_type_logits = type_logits_list[start_index: end_index]
            sample_sel_num_logits = sel_num_logits_list[start_index: end_index]
            sample_where_num_logits = where_num_logits_list[start_index: end_index]
            sample_op_logits_list = op_logits_list[start_index: end_index]

            sample_tag_probs = tag_probs_list[start_index: end_index]
            sample_agg_probs = agg_probs_list[start_index: end_index]
            sample_connection_probs = connection_probs_list[start_index: end_index]
            sample_con_num_probs = con_num_probs_list[start_index: end_index]
            sample_type_probs = type_probs_list[start_index: end_index]
            sample_sel_num_probs = sel_num_probs_list[start_index: end_index]
            sample_where_num_probs = where_num_probs_list[start_index: end_index]
            sample_op_probs = op_probs_list[start_index: end_index]

            cls_index = cls_index_list[start_index]
            table_header_list = table_dict[sample_table_id]["header"]
            table_type_list = table_dict[sample_table_id]["types"]
            table_row_list = table_dict[sample_table_id]["rows"]
            col_dict = {i: [] for i in range(len(table_header_list))}
            for row in table_row_list:
                for col, value in enumerate(row):
                    col_dict[col].append(str(value))
            """
            table_title = table_dict[sample_table_id]["title"]
            table_header_list = table_dict[sample_table_id]["header"]
            table_row_list = table_dict[sample_table_id]["rows"]
            """
            value_change_list = []
            sel_prob_list = []
            where_prob_list = []

            mean_sel_num_prob = np.mean(sample_sel_num_probs, 0)
            mean_where_num_prob = np.mean(sample_where_num_probs, 0)

            for j, col_type in enumerate(sample_type_logits):
                col_data_type = table_type_list[j]
                col_values = col_dict[j]
                type_probs = type_probs_list[j]
                sel_prob = type_probs[0]
                where_prob = type_probs[1]

                # sel
                agg = sample_agg_logits[j]
                sel_col = j
                # agg 与 sel 不匹配的，不进入候选
                if agg in [1, 2, 3, 5] and col_data_type == "text":
                    pass
                else:
                    sel_prob_list.append({"prob": sel_prob, "type": col_type, "sel": sel_col, "agg": agg})

                # where
                tag_list = sample_tag_logits[j][1: cls_index - 1]
                con_num = sample_con_num_logits[j]
                col_op = sample_op_logits_list[j]
                con_col = j
                candidate_list = [[[], []]]
                candidate_list_index = 0
                value_start_index_list = []
                previous_tag = -1

                # 把 token 的 tag_list 扩展成 question 长度
                question_tag_list = []
                for i in range(len(tag_list)):
                    tag = tag_list[i]
                    token = sample_question_token[i]
                    print(token)
                    token = token.replace("##", "")
                    if token == "[UNK]":
                        question_tag_list.extend([tag])
                    else:
                        question_tag_list.extend([tag] * len(token))

                for i in range(0, len(question_tag_list)):
                    current_tag = question_tag_list[i]
                    # 一个 value 结束
                    if current_tag == 0:
                        if previous_tag == 1:
                            candidate_list.append([[], []])
                            candidate_list_index += 1
                    # 一个 value 开始
                    else:
                        if previous_tag in [-1, 0]:
                            value_start_index_list.append(i)
                        candidate_list[candidate_list_index][0].append(sample_question[i])  # 多了一个 cls
                        candidate_list[candidate_list_index][1].append(question_tag_list[i])
                    previous_tag = current_tag
                con_list = []
                # for candidate in candidate_list:
                for i in range(len(value_start_index_list)):
                    candidate = candidate_list[i]
                    value_start_index = value_start_index_list[i]
                    str_list = candidate[0]
                    if len(str_list) == 0: continue
                    value_str = "".join(str_list)
                    op = col_op
                    """
                    if (con_col == 2 and op == 2 and value_str == "1000") or \
                        (con_col == 6 and op == 2 and value_str == "2015年") or \
                        (con_col == 5 and op == 2 and value_str == "350k") or \
                        (con_col == 2 and op == 0 and value_str == "20万") or \
                        (con_col == 6 and op == 2 and value_str == "2016年"):
                        print(1)
                    """
                    candidate_value_set = set()
                    new_value, longest_digit_num, longest_chinese_num = ValueOptimizer.find_longest_num(value_str, sample_question, value_start_index)
                    candidate_value_set.add(value_str)
                    candidate_value_set.add(new_value)
                    if longest_digit_num:
                        candidate_value_set.add(longest_digit_num)
                    digit = None
                    if longest_chinese_num:
                        candidate_value_set.add(longest_chinese_num)
                        digit = ValueOptimizer.chinese2digits(longest_chinese_num)
                        if digit:
                            candidate_value_set.add(digit)
                    replace_candidate_set = ValueOptimizer.create_candidate_set(value_str)
                    candidate_value_set |= replace_candidate_set
                    # 确定 value 值
                    final_value = value_str  # default
                    if op != 2:  # 不是 =，不能搜索，能比大小的应该就是数字
                        if longest_digit_num:
                            final_value = longest_digit_num
                            if final_value != value_str: value_change_list.append([value_str, final_value])
                        elif digit:
                            final_value = digit
                            if final_value != value_str: value_change_list.append([value_str, final_value])
                    else:
                        if value_str not in col_values:
                            best_value = ValueOptimizer.select_best_matched_value_from_candidates(
                                candidate_value_set, col_values)
                            if len(best_value) > 0:
                                final_value = best_value
                                if final_value != value_str: value_change_list.append([value_str, final_value])
                            else:
                                value_change_list.append([value_str, "丢弃"])
                                continue  # =，不在列表内，也没找到模糊匹配，抛弃
                    # 修正"负值"
                    if final_value in ["负值", "负"]:
                        final_value = "0"
                        op = 1
                    if op == 2:
                        # 去除".0"
                        final_value = ValueOptimizer.remove_dot_zero(final_value)
                    elif op in [0, 1]:
                        # 判断 col 类型和 op、value 是否匹配
                        if ValueOptimizer.is_float(final_value) is False or col_data_type == "text": continue
                        # 单位对齐
                        final_value = ValueOptimizer.magnitude_alignment(value_str, value_start_index, sample_question, final_value, col_data_type, col_values)
                    # con_list 是一列里面的 con
                    con_list.append([con_col, op, final_value])
                    """
                    if col_data_type == "text":
                        if value_str not in col_values:
                            best_value, _ = value_optimizer.select_best_matched_value(value_str, col_values)
                            if len(best_value) > 0:
                                value_str = best_value
                    elif col_data_type == "real":
                        if op != 2: # 不是 =，不能搜索，能比大小的应该就是数字
                            if longest_digit_num:
                                value_str = longest_digit_num
                            elif digit:
                                value_str = digit
                    """
                if len(con_list) == con_num:
                    for [con_col, op, final_value] in con_list:
                        where_prob_list.append({"prob": where_prob, "type": col_type, "cond": [con_col, op, final_value]})
                else:
                    if len(con_list) > 0:
                        [con_col, op, final_value] = con_list[0]
                        where_prob_list.append({"prob": where_prob, "type": col_type, "cond": [con_col, op, final_value]})

            # sel_num = max(sample_sel_num_logits, key=sample_sel_num_logits.count)
            # where_num = max(sample_where_num_logits, key=sample_where_num_logits.count)

            # connection = max(real_connection_list, key=real_connection_list.count) if where_num > 1 and len(real_connection_list) > 0 else 0
            # type_dict = {0: "sel", 1: "con", 2: "none"}
            sel_prob_list = sorted(sel_prob_list, key=lambda x: (-x["type"], x["prob"]), reverse=True)
            where_prob_list = sorted(where_prob_list, key=lambda x: (-(x["type"] ** 2 - 1) ** 2, x["prob"]), reverse=True)

            # 联合解码，确定最佳 sel_num 和 where_num
            sel_num = self.determine_num(sel_prob_list, mean_sel_num_prob, num_limit=3)
            where_num = self.determine_num(where_prob_list, mean_where_num_prob, num_limit=4)

            # TODO: connection只有where时才预测，要改过来，前where
            if where_num <= 1 or len(where_prob_list) == 0:
                connection = 0
            else:
                where_cols = list(map(lambda x: x["cond"][0], where_prob_list[: where_num]))
                real_connection_list = [sample_connection_logits[k] for k in where_cols]
                connection = max(real_connection_list, key=real_connection_list.count)

            tmp_sql_dict = copy.deepcopy(sql_dict)
            tmp_sql_dict["cond_conn_op"] = connection
            for j in range(min(sel_num, len(sel_prob_list))):
                tmp_sql_dict["agg"].append(sel_prob_list[j]["agg"])
                tmp_sql_dict["sel"].append(sel_prob_list[j]["sel"])
            # 从 where 候选集中删除已经被 sel 的列
            where_prob_list = list(filter(lambda x: x["cond"][0] not in tmp_sql_dict["sel"], where_prob_list))
            for j in range(min(where_num, len(where_prob_list))):
                tmp_sql_dict["conds"].append(where_prob_list[j]["cond"])
            sql_list.append(tmp_sql_dict)

            if do_test is False:
                if self.sql_match(tmp_sql_dict, sample_sql):
                    matched_num += 1
                else:
                    f_valid.write("%s\n" % str(sample_question))
                    f_valid.write("%s\n" % str(tmp_sql_dict))
                    f_valid.write("%s\n" % str(sample_sql))
                    # f_valid.write("%s\n" % str(value_change_list))
                    cols = set(map(lambda x: x[0], tmp_sql_dict["conds"])) | set(map(lambda x: x[0], sample_sql["conds"]))
                    for j, table_header in enumerate(table_header_list):
                        if j in cols:
                            f_valid.write("%d、%s\n" % (j, table_header))
                    f_valid.write("\n")

        if do_test is False:
            logical_acc = matched_num / len(sample_index_list)
            print("logical_acc", logical_acc)

            op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}
            agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
            conn_sql_dict = {0: "", 1: "and", 2: "or"}
            con_num_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
            type_dict = {0: "sel", 1: "con", 2: "none"}

            tag_pred = []
            tag_true = []
            tag_fully_matched = []
            agg_pred = []
            agg_true = []
            connection_pred = []
            connection_true = []
            con_num_pred = []
            con_num_true = []
            type_pred = type_logits_list
            type_true = type_labels_list
            sel_num_pred = sel_num_logits_list
            sel_num_true = sel_num_labels_list
            where_num_pred = where_num_logits_list
            where_num_true = where_num_labels_list
            op_pred = []
            op_true = []

            for i, col_type in enumerate(type_true):
                if col_type == 0: # sel
                    agg_pred.append(agg_logits_list[i])
                    agg_true.append(agg_labels_list[i])
                elif col_type == 1: # con
                    cls_index = cls_index_list[i]
                    tmp_tag_pred = tag_logits_list[i][1: cls_index - 1] # 不取 cls 和 sep
                    tmp_tag_true = tag_labels_list[i][1: cls_index - 1]
                    question = header_question_list[i]
                    table_id = header_table_id_list[i]
                    matched = 1 if tmp_tag_pred == tmp_tag_true else 0
                    tag_fully_matched.append(matched)
                    tag_pred.extend(tmp_tag_pred)
                    tag_true.extend(tmp_tag_true)
                    connection_pred.append(connection_logits_list[i])
                    connection_true.append(connection_labels_list[i])
                    con_num_pred.append(con_num_logits_list[i])
                    con_num_true.append(con_num_labels_list[i])
                    op_pred.append(op_logits_list[i])
                    op_true.append(op_labels_list[i])

            eval_result = ""
            eval_result += "TYPE\n" + self.detail_score(type_true, type_pred, num_labels=3, ignore_num=None) + "\n"
            eval_result += "TAG\n" + self.detail_score(tag_true, tag_pred, num_labels=2, ignore_num=None) + "\n"
            eval_result += "CONNECTION\n" + self.detail_score(connection_true, connection_pred, num_labels=3, ignore_num=None) + "\n"
            eval_result += "CON_NUM\n" + self.detail_score(con_num_true, con_num_pred, num_labels=4, ignore_num=0) + "\n"
            eval_result += "AGG\n" + self.detail_score(agg_true, agg_pred, num_labels=6, ignore_num=None) + "\n"
            eval_result += "SEL_NUM\n" + self.detail_score(sel_num_true, sel_num_pred, num_labels=4, ignore_num=0) + "\n"
            eval_result += "WHERE_NUM\n" + self.detail_score(where_num_true, where_num_pred, num_labels=5, ignore_num=0) + "\n"
            eval_result += "OP\n" + self.detail_score(op_true, op_pred, num_labels=4, ignore_num=None) + "\n"

            tag_acc = accuracy_score(tag_true, tag_pred)

            return eval_result, tag_acc, logical_acc

        else:
            f_result = open("result.json", 'w')
            for sql_dict in sql_list:
                sql_dict_json = json.dumps(sql_dict, ensure_ascii=False)
                f_result.write(sql_dict_json + '\n')
            f_result.close()

            
    def train(self,model_name = None):
        if self.debug == True:
            # 一次就好
            self.epochs = 1
        train_iterator, valid_iterator,valid_features_labels,valid_table = self.data_iterator()
        print('加载bert')
        lr = 1e-5
        accumulation_steps = math.ceil(self.batch_size / self.base_batch_size)
        # 加载预训练模型
        model = BertNeuralNet.from_pretrained(self.bert_model_path, cache_dir=None)
        model.zero_grad()
        if torch.cuda.is_available():model = model.to(self.device)
        # 不同的参数组设置不同的 weight_decay
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        epoch_steps = int(train_iterator.sampler.num_samples / self.base_batch_size / accumulation_steps)
        num_train_optimization_steps = int(self.epochs * epoch_steps)
        valid_every = math.floor(epoch_steps * accumulation_steps / 5)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps)
        print('开始训练')
        f_log = open("train_log_rewrite.txt", "w")
        best_score = 0
        model.train()
        for epoch in range(self.epochs):
            train_start = time.time()
            optimizer.zero_grad()
            for i , batch_data in enumerate(train_iterator):
                if torch.cuda.is_available():
                    print('epoch:',epoch,'batchIndex:',i)
                    # input_ids = batch_data[0].to(self.device)
                    # sequence_labeling_inputMask = batch_data[1].to(self.device)
                    # sel_column_mask = batch_data[2].to(self.device)
                    # where_conlumn_inputMask = batch_data[3].to(self.device)
                    # type_mask = batch_data[4].to(self.device)
                    # attention_masks = batch_data[5].to(self.device)
                    # where_relation_label = batch_data[6].to(self.device)
                    # sel_agg_label = batch_data[7].to(self.device)
                    # sequence_labeling_label = batch_data[8].to(self.device)
                    # where_conlumn_number_label = batch_data[9].to(self.device)
                    # type_label = batch_data[10].to(self.device)
                    # firstColumn_CLS_startPosition = batch_data[11].to(self.device)
                    # header_mask = batch_data[12].to(self.device)
                    # question_mask = batch_data[13].to(self.device)
                    # nextColumn_CLS_startPosition = batch_data[14].to(self.device)
                    # nextColumn_inputMask = batch_data[15].to(self.device)
                    # sel_num_label = batch_data[16].to(self.device)
                    # where_num_label = batch_data[17].to(self.device)
                    # op_label = batch_data[18].to(self.device)
                    # value_mask = batch_data[19].to(self.device)

                    if torch.sum(batch_data[2].to(self.device)) == 0 or torch.sum(batch_data[3].to(self.device)) == 0 or torch.sum(batch_data[1].to(self.device)) == 0: continue
                    train_dependencies = [
                        batch_data[1].to(self.device),
                        batch_data[2].to(self.device), 
                        batch_data[3].to(self.device),
                        batch_data[6].to(self.device),
                        batch_data[7].to(self.device),
                        batch_data[8].to(self.device),
                        batch_data[9].to(self.device),
                        batch_data[10].to(self.device),
                        batch_data[16].to(self.device), 
                        batch_data[17].to(self.device),
                        batch_data[18].to(self.device)
                        ]
                    loss = model(
                        batch_data[0].to(self.device), 
                        batch_data[5].to(self.device),
                        batch_data[4].to(self.device),
                        batch_data[12].to(self.device), 
                        batch_data[13].to(self.device),
                        batch_data[15].to(self.device),
                        batch_data[14].to(self.device),
                        batch_data[19].to(self.device),
                        batch_data[11].to(self.device), 
                        train_dependencies=train_dependencies)
                    loss.backward()
                    if (i+1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()


            valid_start_time = time.time()
            model.eval()

            tag_logits_list = []
            agg_logits_list = []
            connection_logits_list = []
            con_num_logits_list = []
            type_logits_list = []
            tag_labels_list = []
            agg_labels_list = []
            connection_labels_list = []
            con_num_labels_list = []
            type_labels_list = []
            cls_index_list = []
            sel_num_labels_list = []
            where_num_labels_list = []
            sel_num_logits_list = []
            where_num_logits_list = []
            type_probs_list = []
            op_labels_list = []
            op_logits_list = []
            tag_probs_list = []
            agg_probs_list = []
            connection_probs_list = []
            con_num_probs_list = []
            type_probs_list = []
            sel_num_probs_list = []
            where_num_probs_list = []
            op_probs_list = []
            for j , valid_batch_data in enumerate(valid_iterator):
                if torch.cuda.is_available():

                    input_ids = valid_batch_data[0].to(self.device)
                    tag_masks = valid_batch_data[1].to(self.device)
                    sel_masks = valid_batch_data[2].to(self.device)
                    con_masks = valid_batch_data[3].to(self.device)
                    type_masks = valid_batch_data[4].to(self.device)
                    attention_masks = valid_batch_data[5].to(self.device)
                    connection_labels = valid_batch_data[6].to(self.device)
                    agg_labels = valid_batch_data[7].to(self.device)
                    tag_labels = valid_batch_data[8].to(self.device)
                    con_num_labels = valid_batch_data[9].to(self.device)
                    type_labels = valid_batch_data[10].to(self.device)
                    cls_indices = valid_batch_data[11].to(self.device)
                    header_masks = valid_batch_data[12].to(self.device)
                    question_masks = valid_batch_data[13].to(self.device)
                    subheader_cls_list = valid_batch_data[14].to(self.device)
                    subheader_masks = valid_batch_data[15].to(self.device)
                    sel_num_labels = valid_batch_data[16].to(self.device)
                    where_num_labels = valid_batch_data[17].to(self.device)
                    op_labels = valid_batch_data[18].to(self.device)
                    value_masks = valid_batch_data[19].to(self.device)
                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits, probs_list = model(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks, subheader_cls_list, value_masks, cls_indices)
                
                connection_labels = connection_labels.to('cpu').numpy().tolist()
                agg_labels = agg_labels.to('cpu').numpy().tolist()
                tag_labels = tag_labels.to('cpu').numpy().tolist()
                con_num_labels = con_num_labels.to('cpu').numpy().tolist()
                type_labels = type_labels.to('cpu').numpy().tolist()
                cls_indices = cls_indices.to('cpu').numpy().tolist()
                sel_num_labels = sel_num_labels.to('cpu').numpy().tolist()
                where_num_labels = where_num_labels.to('cpu').numpy().tolist()
                op_labels = op_labels.to('cpu').numpy().tolist()

                tag_logits_list.extend(tag_logits)
                agg_logits_list.extend(agg_logits)
                connection_logits_list.extend(connection_logits)
                con_num_logits_list.extend(con_num_logits)
                type_logits_list.extend(type_logits)
                tag_labels_list.extend(tag_labels)
                agg_labels_list.extend(agg_labels)
                connection_labels_list.extend(connection_labels)
                con_num_labels_list.extend(con_num_labels)
                type_labels_list.extend(type_labels)
                cls_index_list.extend(cls_indices)
                sel_num_labels_list.extend(sel_num_labels)
                where_num_labels_list.extend(where_num_labels)
                sel_num_logits_list.extend(sel_num_logits)
                where_num_logits_list.extend(where_num_logits)
                type_probs_list.extend(type_probs)
                op_labels_list.extend(op_labels)
                op_logits_list.extend(op_logits)

                tag_probs_list.extend(probs_list[0])
                agg_probs_list.extend(probs_list[1])
                connection_probs_list.extend(probs_list[2])
                con_num_probs_list.extend(probs_list[3])
                type_probs_list.extend(probs_list[4])
                sel_num_probs_list.extend(probs_list[5])
                where_num_probs_list.extend(probs_list[6])
                op_probs_list.extend(probs_list[7])

            total_probs_list = [tag_probs_list, agg_probs_list, connection_probs_list, con_num_probs_list, type_probs_list, sel_num_probs_list, where_num_probs_list, op_probs_list]

            logits_lists = [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list, op_logits_list]
            labels_lists = [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list, type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list]
            eval_result, tag_acc, logical_acc = self.evaluate(
                logits_lists, cls_index_list, labels_lists, 
                valid_features_labels.quesion_list, 
                valid_features_labels.question_tokens, 
                valid_features_labels.table_id_list, 
                valid_features_labels.each_trainData_index, 
                valid_features_labels.sql_label, 
                valid_table, 
                valid_features_labels.header_mask, 
                valid_features_labels.table_id_list, total_probs_list)

            score = logical_acc
            print("epoch: %d, train_duration: %d min , valid_duration: %d min \n" % (epoch + 1, int((valid_start_time - train_start_time) / 60), int((time.time() - valid_start_time) / 60)))
            print(eval_result)
            f_log.write("epoch: %d, train_duration: %d min , valid_duration: %d min \n" % (epoch + 1, int((valid_start_time - train_start_time) / 60), int((time.time() - valid_start_time) / 60)))
            f_log.write("\nOVERALL\nlogical_acc: %.3f, tag_acc: %.3f\n\n" % (logical_acc, tag_acc))
            f_log.write(eval_result + "\n")
            f_log.flush()
            save_start_time = time.time()

            if not self.debug and score > best_score:
                best_score = score
                state_dict = model.state_dict()
                # model[bert][seed][epoch][stage][model_name][stage_train_duration][valid_duration][score].bin
                # model_name = "model2/model_%s_%d_%d_%dmin_%dmin_%.4f.bin" % (self.model_name, self.seed, epoch + 1, train_duration, valid_duration, score)
                model_name = "my_model.bin"
                torch.save(state_dict, model_name)
                print("model save duration: %d min" % int((time.time() - save_start_time) / 60))
                f_log.write("model save duration: %d min\n" % int((time.time() - save_start_time) / 60))

            model.train()
        f_log.close()
        # del 训练相关输入和模型
        training_history = [train_iterator, valid_iterator, model, optimizer, param_optimizer, optimizer_grouped_parameters]
        for variable in training_history:
            del variable
        gc.collect()
    
    
    
    def test(self,mode_valid = True,model_predict = False,ensemble = False):
        pass
    def main(self):
        self.train()




    

if __name__ == "__main__":
    data_dir = "./data"
    nl2sql = NL2SQL(data_dir, epochs=15, batch_size=16, base_batch_size=16, max_seq_len=128, debug = True)
    time1 = time.time()
    # nl2sql.data_iterator()
    nl2sql.main()
    print("训练时间: %d min" % int((time.time() - time1) / 60))