import os
import random
import copy
import torch
import json
import time
import math
import gc
import re
import argparse

import numpy as np
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

from utils.config import get_train_logger, timer
from utils.Evaluate import Evaluate
from modules.regex_engine import RegexEngine
from modules.schema_linking import SchemaLiking
from modules.bertnl2sql import BertNL2SQL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class InputFeaturesLabelsForTensor:
    def __init__(self):
        self.connect_inputIDsList = []
        self.sequence_labeling_inputMaskList = []
        self.select_column_inputMaskList = []
        self.where_conlumn_inputMaskList = []
        self.sel_where_detemine_inputMaskList = []
        self.attention_inputMaskList = []
        self.where_relation_labelList = []
        self.select_agg_labelList = []
        self.sequence_labeling_labelList = []
        self.where_column_number_labelList = []
        self.sel_where_detemine_labellist = []
        self.firstColumn_CLS_startPositionList = []
        self.question_list = []
        self.table_id_list = []
        self.eachData_indexList = []
        self.sql_list = []
        self.column_queryList = []
        self.column_tableidList = []
        self.each_column_inputMaskList = []
        self.question_masks = []
        self.nextColumn_CLS_startPositionList = []
        self.nextColumn_inputMaskList = []
        self.select_number_labelList = []
        self.where_number_labelList = []
        self.where_op_labelList = []
        self.value_masks = []
        self.question_token_list = []


class InputFeaturesLabelsForProcess():
    def __init__(self):
        self.connect_inputIDs = []
        self.sequence_labeling_inputMask = []
        self.select_column_inputMask = []
        self.where_conlumn_inputMask = []
        self.sel_where_detemine_inputMask = []
        self.attention_inputMask = []
        self.where_relation_label = []
        self.select_agg_label = []
        self.sequence_labeling_label = []
        self.where_column_number_label = []
        self.sel_where_detemine_label = []
        self.firstColumn_CLS_startPosition = []
        self.question_list = []
        self.table_id_list = []
        self.eachData_index = []
        self.sql_list = []
        self.column_queryList = []
        self.column_tableidList = []
        self.each_column_inputMask = []
        self.question_masks = []
        self.nextColumn_CLS_startPosition = []
        self.nextColumn_inputMask = []
        self.select_number_label = []
        self.where_number_label = []
        self.where_op_label = []
        self.value_masks = []
        self.question_token_list = []


class NL2SQL:
    def __init__(self, config, epochs=1, batch_size=64, step_batch_size=32, max_len=120, seed=1234, debug=False):
        self.device = torch.device('cuda')
        self.config = config
        self.data_dir = config.data_dir
        self.model_dir = config.model_dir
        self.log_dir = config.log_dir
        self.debug = debug
        self.seed = seed
        self.seed_everything()
        self.max_len = max_len
        self.epochs = epochs
        self.step_batch_size = step_batch_size
        self.batch_size = batch_size
        if not os.path.exists(self.data_dir):
            raise NotImplementedError()

        else:
            self.train_data_path = os.path.join(self.data_dir, "train/train.json")
            self.train_table_path = os.path.join(self.data_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(self.data_dir, "val/val.json")
            self.valid_table_path = os.path.join(self.data_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(self.data_dir, "test/test.json")
            self.test_table_path = os.path.join(self.data_dir, "test/test.tables.json")
            self.bert_model_path = os.path.join(self.model_dir, "chinese_wwm_ext_pytorch/")
            self.pytorch_bert_path = os.path.join(self.model_dir, "chinese_wwm_ext_pytorch/pytorch_model.bin")
            self.bert_config = BertConfig(os.path.join(self.model_dir, "chinese_wwm_ext_pytorch/bert_config.json"))

    def read_query(self, query_path):
        """
        query_path 是带有用户问题的json 文件路径
        """
        data = []
        with open(query_path, "r", encoding="utf-8") as data_file:
            for line_index, each_line in enumerate(data_file):
                # debug 只读100行即可
                if self.debug and line_index == 100:
                    break
                data.append(json.loads(each_line))
        logger.info('dataNumber:{}'.format(len(data)))
        return data

    def read_table(self, table_path):
        '''
        table_path 是对应于问题的存有完整数据库的json文件
        '''
        table = {}
        with open(table_path, "r", encoding="utf-8") as table_file:
            for line_index, each_line in enumerate(table_file):
                each_table = json.loads(each_line)
                table[each_table['id']] = each_table
        return table

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        # 固定随机数的种子保证结果可复现性
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def create_mask(self, max_len, start_index, mask_len):
        '''
        对给定的序列中返回他对应的 mask 序列
        只保留起始索引到mask 长度的序列为1 ，其余为0
        '''
        mask = [0] * max_len
        for i in range(start_index, start_index + mask_len):
            mask[i] = 1
        return mask

    def process_sample(self, query, table, bert_tokenizer):
        question = query["question"]
        tableID = query["table_id"]
        select_column = query["sql"]["sel"]
        select_agg = query["sql"]["agg"]
        where_conditions = query["sql"]["conds"]
        where_relation = query["sql"]["cond_conn_op"]
        '''
        table[tableID]
        {'rows': [['死侍2：我爱我家', 10637.3, 25.8, 5.0], ['白蛇：缘起', 10503.8, 25.4, 7.0], ['大黄蜂', 6426.6, 15.6, 6.0], ['密室逃生', 5841.4, 14.2, 6.0], ['“大”人物', 3322.9, 8.1, 5.0], ['家和万事惊', 635.2, 1.5, 25.0], ['钢铁飞龙之奥特曼崛起', 595.5, 1.4, 3.0], ['海王', 500.3, 1.2, 5.0], ['一条狗的回家路', 360.0, 0.9, 4.0], ['掠食城市', 356.6, 0.9, 3.0]], 'name': 'Table_4d29d0513aaa11e9b911f40f24344a08', 'title': '表3：2019年第4周（2019.01.28 - 2019.02.03）全国电影票房TOP10', 'header': ['影片名称', '周票房（万）', '票房占比（%）', '场均人次'], 
        'common': '资料来源：艺恩电影智库，光大证券研究所', 'id': '4d29d0513aaa11e9b911f40f24344a08', 'types': ['text', 'real', 'real', 'real']}
        '''
        header_list = table[tableID]["header"]
        row_list = table[tableID]["rows"]
        '''
        row_list: 是数据库中具体存放的行列
        [['死侍2：我爱我家', 10637.3, 25.8, 5.0], ['白蛇：缘起', 10503.8, 25.4, 7.0], ['大黄蜂', 6426.6, 15.6, 6.0], ['密室逃生', 5841.4, 14.2, 6.0], ['“大”人物', 3322.9, 8.1, 5.0], ['家和万事惊', 635.2, 1.5, 25.0], ['钢铁飞龙之奥特曼崛起', 595.5, 1.4, 3.0], ['海王', 500.3, 1.2, 5.0], ['一条狗的回家路', 360.0, 0.9, 4.0], ['掠食城市', 356.6, 0.9, 3.0]]
        '''
        # 去除空格和换行等
        question = question.strip().replace(" ", "")
        columnValue_dict = {each_column: set() for each_column in header_list}
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
        duplicate_indices = SchemaLiking.duplicate_relative_index(where_conditions)
        condition_dict = {}
        for [where_col, where_op, where_value], duplicate_index in zip(where_conditions, duplicate_indices):
            where_value = where_value.strip()
            matched_value, matched_index = SchemaLiking.match_value(question, where_value, duplicate_index)
            '''
            question                二零一九年第四周大黄蜂和密室逃生这两部影片的票房总占比是多少呀
            matched_value           大黄蜂
            match_index             8
            '''

            if len(matched_value) > 0:
                if where_col in condition_dict:
                    condition_dict[where_col].append([where_op, matched_value, matched_index])
                else:
                    condition_dict[where_col] = [[where_op, matched_value, matched_index]]
            else:
                # TODO 是否存在匹配不到值得情况，以及 该如何处理
                pass
            # condition_dict : {0: [[2, '大黄蜂', 8]]}
        inputFeaturesLabelsForProcess = InputFeaturesLabelsForProcess()

        question_ = bert_tokenizer.tokenize(question)
        question_UNK_position = []
        for index, each_token in enumerate(question_):
            each_token = each_token.replace("##", "")
            if each_token == "[UNK]":
                # TODO 似乎不存在 [UNK]
                question_UNK_position.extend([index])
                # print(question)
                # exit()

            else:
                question_UNK_position.extend([index] * len(each_token))
        question_inputIDs = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + question_ + ["[SEP]"])
        firstColumn_CLS_startPosition = len(question_inputIDs)
        question_inputMask = self.create_mask(max_len=self.max_len, start_index=1, mask_len=len(question_))
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
            each_column_inputMask = self.create_mask(max_len=self.max_len, start_index=len(question_inputIDs) + 1,
                                                     mask_len=len(each_column_))

            connect_inputIDs = question_inputIDs + each_column_inputIDs
            # question + column 后面再接的column对应的CLS的索引
            nextColumn_CLS_startPosition = len(connect_inputIDs)
            # 后面的column的 起始索引
            nextColumn_startPosition = nextColumn_CLS_startPosition + 1
            random.seed(index_header)
            for index_nextColumn, nextColumn in enumerate(random.sample(header_list, len(header_list))):
                nextColumn_ = bert_tokenizer.tokenize(nextColumn)
                if index_nextColumn == 0:
                    nextColumn_inputIDs = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + nextColumn_ + ["[SEP]"])
                else:
                    nextColumn_inputIDs = bert_tokenizer.convert_tokens_to_ids(nextColumn_ + ["[SEP]"])
                if len(connect_inputIDs) + len(nextColumn_inputIDs) <= self.max_len:
                    connect_inputIDs += nextColumn_inputIDs
                else:
                    break
            # nextColumn_inputMask_len 要mask掉的后面的列的长度
            nextColumn_inputMask_len = len(connect_inputIDs) - nextColumn_startPosition - 1
            nextColumn_inputMask = self.create_mask(max_len=self.max_len, start_index=nextColumn_startPosition,
                                                    mask_len=nextColumn_inputMask_len)
            '''
            nextColumn_inputMask 
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            '''
            # 这里的起始位置指的是connect_id中的索引
            value_CLS_startPosition = len(connect_inputIDs)
            value_startPosition = len(connect_inputIDs) + 1
            # print(value_dict)
            for value_index, each_value in enumerate(value_dict):
                each_value_ = bert_tokenizer.tokenize(each_value)
                if value_index == 0:
                    value_inputIDs = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + each_value_ + ["[SEP]"])
                else:
                    value_inputIDs = bert_tokenizer.convert_tokens_to_ids(each_value_ + ["[SEP]"])
                if len(connect_inputIDs) + len(value_inputIDs) <= self.max_len:
                    connect_inputIDs += value_inputIDs
                else:
                    break
            value_inputMask_len = len(connect_inputIDs) - value_startPosition - 1
            value_inputMask = self.create_mask(max_len=self.max_len, start_index=value_startPosition,
                                               mask_len=value_inputMask_len)
            # 此时connect_inputIDs 相当于 CLS + query+ SEP +column1+ SEP+ nextcolumn1 +SEP + nextcolumn2+ SEP + value1 + SEP
            # value 对应的是数据库里当前header 下面的全部的value
            # attention_mask 是 对当前是connet_inputIDs 做mask            
            attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(connect_inputIDs))
            # padding 
            connect_inputIDs = connect_inputIDs + [0] * (self.max_len - len(connect_inputIDs))

            sequence_labeling_label = [0] * len(connect_inputIDs)
            select_column_mask, where_conlumn_inputMask, type_mask = 0, 0, 1
            # TODO op_label 一直都是2是不是有问题？
            where_relation_label, select_agg_label, where_conlumn_number_label, op_label = 0, 0, 0, 2
            '''
            condition_dict 
            {0: [[2, '大黄蜂', 8], [2, '密室逃生', 12]]}
            '''
            if index_header in condition_dict:
                # TODO 这地方是不是可以优化一下
                if list(map(lambda x: x[0], where_conditions)).count(index_header) != len(
                        condition_dict[index_header]): continue
                conlumn_condition_list = condition_dict[index_header]
                for [conlumn_condition_op, conlumn_condition_value, conlumn_condition_index] in conlumn_condition_list:
                    value_startposition_inQuestion = conlumn_condition_index
                    # end_position : 8+len('大黄蜂') -1 = 10
                    value_endposition_inQuestion = conlumn_condition_index + len(conlumn_condition_value) - 1
                    # 处理了一下UNK
                    value_startposition_forLabeling = question_UNK_position[value_startposition_inQuestion] + 1  # cls
                    value_endposition_forLabeling = question_UNK_position[
                                                        value_endposition_inQuestion] + 1 + 1  # cls sep
                    # 序列标注将问题question中value对应的部分标注成1
                    sequence_labeling_label[value_startposition_forLabeling:value_endposition_forLabeling] = [1] * (
                            value_endposition_forLabeling - value_startposition_forLabeling)
                    # TODO 序列标注inputID 是问题中的value ,inpustMask是整个问题？ 
                sequence_labeling_inputMask = [0] + [1] * len(question_) + [0] * (self.max_len - len(question_) - 1)
                where_conlumn_inputMask = 1
                where_relation_label = where_relation
                where_conlumn_number_label = min(len(conlumn_condition_list), 3)
                type_label = 1
            elif index_header in select_clause_dict:
                sequence_labeling_inputMask = [0] * self.max_len
                select_column_mask = 1
                select_agg_label = select_clause_dict[index_header]
                type_label = 0
            else:
                sequence_labeling_inputMask = [0] * self.max_len
                type_label = 2
            '''
            这里相当于挨个遍历header_list中的列然后依次给对应的变量打上标签
            如果当前的列在condition_dict 也就是在conds 中，那么给对应的问题打上序列标注的标签
            type_label 是用来空值当前的列对应的是 sel的列还是 where 里的列，如果标记为2 那么就表示不选择当前的这个列
            '''
            inputFeaturesLabelsForProcess.connect_inputIDs.append(connect_inputIDs)
            inputFeaturesLabelsForProcess.sequence_labeling_inputMask.append(sequence_labeling_inputMask)
            inputFeaturesLabelsForProcess.select_column_inputMask.append(select_column_mask)
            inputFeaturesLabelsForProcess.where_conlumn_inputMask.append(where_conlumn_inputMask)
            inputFeaturesLabelsForProcess.sel_where_detemine_inputMask.append(type_mask)
            inputFeaturesLabelsForProcess.attention_inputMask.append(attention_mask)
            inputFeaturesLabelsForProcess.where_relation_label.append(where_relation_label)
            inputFeaturesLabelsForProcess.select_agg_label.append(select_agg_label)
            inputFeaturesLabelsForProcess.sequence_labeling_label.append(sequence_labeling_label)
            inputFeaturesLabelsForProcess.where_column_number_label.append(where_conlumn_number_label)
            inputFeaturesLabelsForProcess.sel_where_detemine_label.append(type_label)
            inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition.append(firstColumn_CLS_startPosition)
            inputFeaturesLabelsForProcess.column_queryList.append(question)
            inputFeaturesLabelsForProcess.column_tableidList.append(tableID)
            inputFeaturesLabelsForProcess.each_column_inputMask.append(each_column_inputMask)
            inputFeaturesLabelsForProcess.question_masks.append(question_inputMask)
            inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition.append(nextColumn_CLS_startPosition)
            inputFeaturesLabelsForProcess.nextColumn_inputMask.append(nextColumn_inputMask)
            inputFeaturesLabelsForProcess.select_number_label.append(select_number_label)
            inputFeaturesLabelsForProcess.where_number_label.append(where_number_label)
            inputFeaturesLabelsForProcess.where_op_label.append(op_label)
            inputFeaturesLabelsForProcess.value_masks.append(value_inputMask)
        return inputFeaturesLabelsForProcess, question_

    def data_process(self, sample, table_dict, bert_tokenizer):
        question = sample["question"]
        table_id = sample["table_id"]
        sel_list = sample["sql"]["sel"]
        agg_list = sample["sql"]["agg"]
        con_list = sample["sql"]["conds"]
        connection = sample["sql"]["cond_conn_op"]
        table_title = table_dict[table_id]["title"]
        table_header_list = table_dict[table_id]["header"]
        table_row_list = table_dict[table_id]["rows"]
        # 去除空格和换行等
        question = question.strip().replace(" ", "")

        col_dict = {header_name: set() for header_name in table_header_list}
        for row in table_row_list:
            for col, value in enumerate(row):
                header_name = table_header_list[col]
                col_dict[header_name].add(str(value))

        sel_num = len(sel_list)
        where_num = len(con_list)
        sel_dict = {sel: agg for sel, agg in zip(sel_list, agg_list)}
        # <class 'list'>: [[0, 2, '大黄蜂'], [0, 2, '密室逃生']] 一列两value 多一个任务判断where一列的value数, con_dict里面的数量要喝conds匹配，否则放弃这一列（但也不能作为非con非sel训练）
        # 标注只能用多分类？有可能对应多个
        duplicate_indices = SchemaLiking.duplicate_relative_index(con_list)
        con_dict = {}
        for [con_col, op, value], duplicate_index in zip(con_list, duplicate_indices):  # duplicate index 是跟着 value 的
            value = value.strip()
            matched_value, matched_index = SchemaLiking.match_value(question, value, duplicate_index)
            if len(matched_value) > 0:
                if con_col in con_dict:
                    con_dict[con_col].append([op, matched_value, matched_index])
                else:
                    con_dict[con_col] = [[op, matched_value, matched_index]]
        # TODO：con_dict要看看len和conds里同一列的数量是否一致，不一致不参与训练
        # TODO：多任务加上col对应的con数量
        # TODO：需要变成训练集的就是 sel_dict、con_dict和connection
        # TODO: 只有conds的序列标注任务是valid的，其他都不valid

        conc_tokens = []
        tag_masks = []
        sel_masks = []
        con_masks = []
        type_masks = []
        attention_masks = []
        header_masks = []
        question_masks = []
        value_masks = []
        connection_labels = []
        agg_labels = []
        tag_labels = []
        con_num_labels = []
        type_labels = []
        cls_index_list = []
        header_question_list = []
        header_table_id_list = []
        subheader_cls_list = []
        subheader_masks = []
        sel_num_labels = []
        where_num_labels = []
        op_labels = []

        question_tokens = bert_tokenizer.tokenize(question)
        question_list = list(question)

        question_char_mapping = []
        for i, token in enumerate(question_tokens):
            token = token.replace("##", "")
            if token == "[UNK]":
                question_char_mapping.extend([i])
            else:
                question_char_mapping.extend([i] * len(token))

        question_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"])
        header_cls_index = len(question_ids)
        question_mask = self.create_mask(max_len=self.max_len, start_index=1, mask_len=len(question_tokens))
        # tag_list = sample_tag_logits[j][1: cls_index - 1]
        for col in range(len(table_header_list)):
            header = table_header_list[col]
            value_set = col_dict[header]
            header_tokens = bert_tokenizer.tokenize(header)
            header_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + header_tokens + ["[SEP]"])
            header_mask = self.create_mask(max_len=self.max_len, start_index=len(question_ids) + 1,
                                           mask_len=len(header_tokens))

            conc_ids = question_ids + header_ids
            subheader_cls_index = len(conc_ids)
            subheader_start_index = len(conc_ids) + 1
            random.seed(col)
            for i, sub_header in enumerate(random.sample(table_header_list, len(table_header_list))):
                subheader_tokens = bert_tokenizer.tokenize(sub_header)
                if i == 0:
                    subheader_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + subheader_tokens + ["[SEP]"])
                else:
                    subheader_ids = bert_tokenizer.convert_tokens_to_ids(subheader_tokens + ["[SEP]"])
                if len(conc_ids) + len(subheader_ids) <= self.max_len:
                    conc_ids += subheader_ids
                else:
                    break
            subheader_mask_len = len(conc_ids) - subheader_start_index - 1
            subheader_mask = self.create_mask(max_len=self.max_len, start_index=subheader_start_index,
                                              mask_len=subheader_mask_len)
            # attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(conc_ids))
            # conc_ids = conc_ids + [0] * (self.max_len - len(conc_ids))

            value_cls_index = len(conc_ids)
            value_start_index = len(conc_ids) + 1
            for i, value in enumerate(value_set):
                value_tokens = bert_tokenizer.tokenize(value)
                if i == 0:
                    value_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + value_tokens + ["[SEP]"])
                else:
                    value_ids = bert_tokenizer.convert_tokens_to_ids(value_tokens + ["[SEP]"])
                if len(conc_ids) + len(value_ids) <= self.max_len:
                    conc_ids += value_ids
                else:
                    break
            value_mask_len = len(conc_ids) - value_start_index - 1
            value_mask = self.create_mask(max_len=self.max_len, start_index=value_start_index, mask_len=value_mask_len)
            attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(conc_ids))
            conc_ids = conc_ids + [0] * (self.max_len - len(conc_ids))

            tag_ids = [0] * len(conc_ids)
            sel_mask, con_mask, type_mask = 0, 0, 1
            connection_id, agg_id, con_num, op = 0, 0, 0, 2
            if col in con_dict:
                # 如果 header 对应多个 values，values 必须全部匹配上才进入训练
                if list(map(lambda x: x[0], con_list)).count(col) != len(con_dict[col]): continue
                header_con_list = con_dict[col]
                for [op, value, index] in header_con_list:
                    value_char_start_index = index
                    value_char_end_index = index + len(value) - 1
                    value_id_start_index = question_char_mapping[value_char_start_index] + 1
                    value_id_end_index = question_char_mapping[
                                             value_char_end_index] + 1 + 1  # 一个1是cls，一个1是end_index回缩一格
                    tag_ids[value_id_start_index: value_id_end_index] = [1] * (
                            value_id_end_index - value_id_start_index)
                tag_mask = [0] + [1] * len(question_tokens) + [0] * (self.max_len - len(question_tokens) - 1)
                con_mask = 1
                connection_id = connection
                con_num = min(len(header_con_list), 3)  # 4 只有一个样本，太少了，归到 3 类
                type_id = 1
            elif col in sel_dict:
                # TODO: 是不是还有同一个个sel col，多个不同聚合方式
                tag_mask = [0] * self.max_len
                sel_mask = 1
                agg_id = sel_dict[col]
                type_id = 0
            else:
                tag_mask = [0] * self.max_len
                type_id = 2
            conc_tokens.append(conc_ids)
            tag_masks.append(tag_mask)
            sel_masks.append(sel_mask)
            con_masks.append(con_mask)
            type_masks.append(type_mask)
            attention_masks.append(attention_mask)
            connection_labels.append(connection_id)
            agg_labels.append(agg_id)
            tag_labels.append(tag_ids)
            con_num_labels.append(con_num)
            type_labels.append(type_id)
            cls_index_list.append(header_cls_index)
            header_question_list.append(question)
            header_table_id_list.append(table_id)
            header_masks.append(header_mask)
            question_masks.append(question_mask)
            subheader_cls_list.append(subheader_cls_index)
            subheader_masks.append(subheader_mask)
            sel_num_labels.append(sel_num)
            where_num_labels.append(where_num)
            op_labels.append(op)
            value_masks.append(value_mask)

        return tag_masks, sel_masks, con_masks, type_masks, attention_masks, connection_labels, agg_labels, tag_labels, con_num_labels, type_labels, cls_index_list, conc_tokens, header_question_list, header_table_id_list, header_masks, question_masks, subheader_cls_list, subheader_masks, sel_num_labels, where_num_labels, op_labels, value_masks, question_tokens

    def process_sample_test(self, sample, tableData, bert_tokenizer):

        # 相关变量的含义参见上面 data_process
        question = sample["question"]
        table_id = sample["table_id"]
        table_title = tableData[table_id]["title"]
        table_header_list = tableData[table_id]["header"]
        table_row_list = tableData[table_id]["rows"]
        question = question.strip().replace(" ", "")

        col_dict = {header_name: set() for header_name in table_header_list}
        for row in table_row_list:
            for col, value in enumerate(row):
                header_name = table_header_list[col]
                col_dict[header_name].add(str(value))

        inputFeaturesLabelsForProcess = InputFeaturesLabelsForProcess()

        question_tokens = bert_tokenizer.tokenize(question)
        question_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"])
        header_cls_index = len(question_ids)
        question_mask = self.create_mask(max_len=self.max_len, start_index=1, mask_len=len(question_tokens))
        for col in range(len(table_header_list)):
            header = table_header_list[col]
            value_set = col_dict[header]
            header_tokens = bert_tokenizer.tokenize(header)
            header_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + header_tokens + ["[SEP]"])
            header_mask = self.create_mask(max_len=self.max_len, start_index=len(question_ids) + 1,
                                           mask_len=len(header_tokens))

            connect_inputIDs = question_ids + header_ids
            subheader_cls_index = len(connect_inputIDs)
            subheader_start_index = len(connect_inputIDs) + 1
            random.seed(col)
            for i, sub_header in enumerate(random.sample(table_header_list, len(table_header_list))):
                subheader_tokens = bert_tokenizer.tokenize(sub_header)
                if i == 0:
                    subheader_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + subheader_tokens + ["[SEP]"])
                else:
                    subheader_ids = bert_tokenizer.convert_tokens_to_ids(subheader_tokens + ["[SEP]"])
                if len(connect_inputIDs) + len(subheader_ids) <= self.max_len:
                    connect_inputIDs += subheader_ids
                else:
                    break
            subheader_mask_len = len(connect_inputIDs) - subheader_start_index - 1
            subheader_mask = self.create_mask(max_len=self.max_len, start_index=subheader_start_index,
                                              mask_len=subheader_mask_len)

            value_cls_index = len(connect_inputIDs)
            value_start_index = len(connect_inputIDs) + 1
            for i, value in enumerate(value_set):
                value_tokens = bert_tokenizer.tokenize(value)
                if i == 0:
                    value_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + value_tokens + ["[SEP]"])
                else:
                    value_ids = bert_tokenizer.convert_tokens_to_ids(value_tokens + ["[SEP]"])
                if len(connect_inputIDs) + len(value_ids) <= self.max_len:
                    connect_inputIDs += value_ids
                else:
                    break
            value_mask_len = len(connect_inputIDs) - value_start_index - 1
            value_mask = self.create_mask(max_len=self.max_len, start_index=value_start_index, mask_len=value_mask_len)
            attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(connect_inputIDs))
            connect_inputIDs = connect_inputIDs + [0] * (self.max_len - len(connect_inputIDs))

            inputFeaturesLabelsForProcess.connect_inputIDs.append(connect_inputIDs)
            inputFeaturesLabelsForProcess.attention_inputMask.append(attention_mask)
            inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition.append(header_cls_index)
            inputFeaturesLabelsForProcess.each_column_inputMask.append(header_mask)
            inputFeaturesLabelsForProcess.question_masks.append(question_mask)
            inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition.append(subheader_cls_index)
            inputFeaturesLabelsForProcess.nextColumn_inputMask.append(subheader_mask)
            inputFeaturesLabelsForProcess.value_masks.append(value_mask)
        inputFeaturesLabelsForProcess.sel_where_detemine_inputMask = [1] * len(
            inputFeaturesLabelsForProcess.connect_inputIDs)

        return inputFeaturesLabelsForProcess, question_tokens

    def data_iterator(self, mode='train'):
        # train: 41522 val: 4396 test: 4086
        if mode == 'train':
            logger.info("Loading data, the mode is \"train\", Loading train_set and valid_set")
            train_data_list = self.read_query(self.train_data_path)
            train_table_dict = self.read_table(self.train_table_path)
            valid_data_list = self.read_query(self.valid_data_path)
            valid_table_dict = self.read_table(self.valid_table_path)
            test_data_list = self.read_query(self.test_data_path)
            test_table_dict = self.read_table(self.test_table_path)
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
            train_conc_tokens = []
            train_tag_masks = []
            train_sel_masks = []
            train_con_masks = []
            train_type_masks = []
            train_attention_masks = []
            train_connection_labels = []
            train_agg_labels = []
            train_tag_labels = []
            train_con_num_labels = []
            train_type_labels = []
            train_cls_index_list = []
            train_question_list = []
            train_table_id_list = []
            train_sample_index_list = []
            train_sql_list = []
            train_header_question_list = []
            train_header_table_id_list = []
            train_header_masks = []
            train_question_masks = []
            train_subheader_cls_list = []
            train_subheader_masks = []
            train_sel_num_labels = []
            train_where_num_labels = []
            train_op_labels = []
            train_value_masks = []
            train_question_token_list = []
            for sample in train_data_list:
                processed_result = self.data_process(sample, train_table_dict, bert_tokenizer)
                train_tag_masks.extend(processed_result[0])
                train_sel_masks.extend(processed_result[1])
                train_con_masks.extend(processed_result[2])
                train_type_masks.extend(processed_result[3])
                train_attention_masks.extend(processed_result[4])
                train_connection_labels.extend(processed_result[5])
                train_agg_labels.extend(processed_result[6])
                train_tag_labels.extend(processed_result[7])
                train_con_num_labels.extend(processed_result[8])
                train_type_labels.extend(processed_result[9])
                train_cls_index_list.extend(processed_result[10])
                train_conc_tokens.extend(processed_result[11])
                train_header_question_list.extend(processed_result[12])
                train_header_table_id_list.extend(processed_result[13])
                train_header_masks.extend(processed_result[14])
                train_question_masks.extend(processed_result[15])
                train_subheader_cls_list.extend(processed_result[16])
                train_subheader_masks.extend(processed_result[17])
                train_sel_num_labels.extend(processed_result[18])
                train_where_num_labels.extend(processed_result[19])
                train_op_labels.extend(processed_result[20])
                train_value_masks.extend(processed_result[21])
                train_question_token_list.append(processed_result[22])
                train_sample_index_list.append(len(train_conc_tokens))
                train_sql_list.append(sample["sql"])
                train_question_list.append(sample["question"].strip().replace(" ", ""))
                train_table_id_list.append(sample["table_id"])
            valid_conc_tokens = []
            valid_tag_masks = []
            valid_sel_masks = []
            valid_con_masks = []
            valid_type_masks = []
            valid_attention_masks = []
            valid_connection_labels = []
            valid_agg_labels = []
            valid_tag_labels = []
            valid_con_num_labels = []
            valid_type_labels = []
            valid_cls_index_list = []
            valid_question_list = []
            valid_table_id_list = []
            valid_sample_index_list = []
            valid_sql_list = []
            valid_header_question_list = []
            valid_header_table_id_list = []
            valid_header_masks = []
            valid_question_masks = []
            valid_subheader_cls_list = []
            valid_subheader_masks = []
            valid_sel_num_labels = []
            valid_where_num_labels = []
            valid_op_labels = []
            valid_value_masks = []
            valid_question_token_list = []
            for sample in valid_data_list:
                processed_result = self.data_process(sample, valid_table_dict, bert_tokenizer)
                valid_tag_masks.extend(processed_result[0])
                valid_sel_masks.extend(processed_result[1])
                valid_con_masks.extend(processed_result[2])
                valid_type_masks.extend(processed_result[3])
                valid_attention_masks.extend(processed_result[4])
                valid_connection_labels.extend(processed_result[5])
                valid_agg_labels.extend(processed_result[6])
                valid_tag_labels.extend(processed_result[7])
                valid_con_num_labels.extend(processed_result[8])
                valid_type_labels.extend(processed_result[9])
                valid_cls_index_list.extend(processed_result[10])
                valid_conc_tokens.extend(processed_result[11])
                valid_header_question_list.extend(processed_result[12])
                valid_header_table_id_list.extend(processed_result[13])
                valid_header_masks.extend(processed_result[14])
                valid_question_masks.extend(processed_result[15])
                valid_subheader_cls_list.extend(processed_result[16])
                valid_subheader_masks.extend(processed_result[17])
                valid_sel_num_labels.extend(processed_result[18])
                valid_where_num_labels.extend(processed_result[19])
                valid_op_labels.extend(processed_result[20])
                valid_value_masks.extend(processed_result[21])
                valid_question_token_list.append(processed_result[22])
                valid_sample_index_list.append(len(valid_conc_tokens))
                valid_sql_list.append(sample["sql"])
                valid_question_list.append(sample["question"].strip().replace(" ", ""))
                valid_table_id_list.append(sample["table_id"])
            train_dataset = data.TensorDataset(torch.tensor(train_conc_tokens, dtype=torch.long),
                                               torch.tensor(train_tag_masks, dtype=torch.long),
                                               torch.tensor(train_sel_masks, dtype=torch.long),
                                               torch.tensor(train_con_masks, dtype=torch.long),
                                               torch.tensor(train_type_masks, dtype=torch.long),
                                               torch.tensor(train_attention_masks, dtype=torch.long),
                                               torch.tensor(train_connection_labels, dtype=torch.long),
                                               torch.tensor(train_agg_labels, dtype=torch.long),
                                               torch.tensor(train_tag_labels, dtype=torch.long),
                                               torch.tensor(train_con_num_labels, dtype=torch.long),
                                               torch.tensor(train_type_labels, dtype=torch.long),
                                               torch.tensor(train_cls_index_list, dtype=torch.long),
                                               torch.tensor(train_header_masks, dtype=torch.long),
                                               torch.tensor(train_question_masks, dtype=torch.long),
                                               torch.tensor(train_subheader_cls_list, dtype=torch.long),
                                               torch.tensor(train_subheader_masks, dtype=torch.long),
                                               torch.tensor(train_sel_num_labels, dtype=torch.long),
                                               torch.tensor(train_where_num_labels, dtype=torch.long),
                                               torch.tensor(train_op_labels, dtype=torch.long),
                                               torch.tensor(train_value_masks, dtype=torch.long)
                                               )
            valid_dataset = data.TensorDataset(torch.tensor(valid_conc_tokens, dtype=torch.long),
                                               torch.tensor(valid_tag_masks, dtype=torch.long),
                                               torch.tensor(valid_sel_masks, dtype=torch.long),
                                               torch.tensor(valid_con_masks, dtype=torch.long),
                                               torch.tensor(valid_type_masks, dtype=torch.long),
                                               torch.tensor(valid_attention_masks, dtype=torch.long),
                                               torch.tensor(valid_connection_labels, dtype=torch.long),
                                               torch.tensor(valid_agg_labels, dtype=torch.long),
                                               torch.tensor(valid_tag_labels, dtype=torch.long),
                                               torch.tensor(valid_con_num_labels, dtype=torch.long),
                                               torch.tensor(valid_type_labels, dtype=torch.long),
                                               torch.tensor(valid_cls_index_list, dtype=torch.long),
                                               torch.tensor(valid_header_masks, dtype=torch.long),
                                               torch.tensor(valid_question_masks, dtype=torch.long),
                                               torch.tensor(valid_subheader_cls_list, dtype=torch.long),
                                               torch.tensor(valid_subheader_masks, dtype=torch.long),
                                               torch.tensor(valid_sel_num_labels, dtype=torch.long),
                                               torch.tensor(valid_where_num_labels, dtype=torch.long),
                                               torch.tensor(valid_op_labels, dtype=torch.long),
                                               torch.tensor(valid_value_masks, dtype=torch.long)
                                               )
            # 将 dataset 转成 dataloader
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.step_batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.step_batch_size, shuffle=False)
            # 返回训练数据
            return train_loader, valid_loader, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list, valid_question_token_list
        elif mode == "test":
            logger.info("Loading data, the mode is \"test\", Loading test_set")
            test_data_list = self.read_query(self.test_data_path)
            test_tableData = self.read_table(self.test_table_path)
            TestFeaturesLabels = InputFeaturesLabelsForTensor()
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
            TestFeaturesLabels = InputFeaturesLabelsForTensor()

            for sample in test_data_list:
                inputFeaturesLabelsForProcess, test_question_tokens = self.process_sample_test(sample, test_tableData,
                                                                                               bert_tokenizer)
                TestFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                TestFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                TestFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                TestFeaturesLabels.each_column_inputMaskList.extend(inputFeaturesLabelsForProcess.each_column_inputMask)
                TestFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                TestFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                TestFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                TestFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                TestFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                TestFeaturesLabels.question_token_list.append(test_question_tokens)
                TestFeaturesLabels.eachData_indexList.append(len(TestFeaturesLabels.connect_inputIDsList))
                TestFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TestFeaturesLabels.table_id_list.append(sample["table_id"])

            test_dataset = data.TensorDataset(torch.tensor(TestFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.attention_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.firstColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.each_column_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.question_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.value_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.sel_where_detemine_inputMaskList,
                                                           dtype=torch.long),
                                              )

            test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.step_batch_size, shuffle=False)

            return test_iterator, TestFeaturesLabels, test_tableData
        elif mode == "evaluate":
            logger.info("Loading data, the mode is \"evaluate\", Loading valid_set")
            valid_data_list = self.read_query(self.valid_data_path)
            valid_tableData = self.read_table(self.valid_table_path)

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            ValidFeaturesLabels = InputFeaturesLabelsForTensor()

            for sample in valid_data_list:
                inputFeaturesLabelsForProcess, valid_question_tokens = self.process_sample(sample, valid_tableData,
                                                                                           bert_tokenizer)
                ValidFeaturesLabels.sequence_labeling_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_inputMask)
                ValidFeaturesLabels.select_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.select_column_inputMask)
                ValidFeaturesLabels.where_conlumn_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.where_conlumn_inputMask)
                ValidFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                ValidFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                ValidFeaturesLabels.where_relation_labelList.extend(inputFeaturesLabelsForProcess.where_relation_label)
                ValidFeaturesLabels.select_agg_labelList.extend(inputFeaturesLabelsForProcess.select_agg_label)
                ValidFeaturesLabels.sequence_labeling_labelList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_label)
                ValidFeaturesLabels.where_column_number_labelList.extend(
                    inputFeaturesLabelsForProcess.where_column_number_label)
                ValidFeaturesLabels.sel_where_detemine_labellist.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_label)
                ValidFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                ValidFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                ValidFeaturesLabels.column_queryList.extend(inputFeaturesLabelsForProcess.column_queryList)
                ValidFeaturesLabels.column_tableidList.extend(inputFeaturesLabelsForProcess.column_tableidList)
                ValidFeaturesLabels.each_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.each_column_inputMask)
                ValidFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                ValidFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                ValidFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                ValidFeaturesLabels.select_number_labelList.extend(inputFeaturesLabelsForProcess.select_number_label)
                ValidFeaturesLabels.where_number_labelList.extend(inputFeaturesLabelsForProcess.where_number_label)
                ValidFeaturesLabels.where_op_labelList.extend(inputFeaturesLabelsForProcess.where_op_label)
                ValidFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question_tokens)
                ValidFeaturesLabels.eachData_indexList.append(len(ValidFeaturesLabels.connect_inputIDsList))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_conlumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_relation_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_agg_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_column_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_labellist,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.firstColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.each_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_op_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )

            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.step_batch_size, shuffle=False)

            return valid_iterator, ValidFeaturesLabels, valid_tableData  # , test_iterator, TestFeaturesLabels, test_tableData
        elif mode == "test&evaluate":
            logger.info("Loading data, the mode is \"test&evaluate\", Loading test_set and valid_set")
            test_data_list = self.read_query(self.test_data_path)
            test_tableData = self.read_table(self.test_table_path)
            valid_data_list = self.read_query(self.valid_data_path)
            valid_tableData = self.read_table(self.valid_table_path)

            TestFeaturesLabels = InputFeaturesLabelsForTensor()
            ValidFeaturesLabels = InputFeaturesLabelsForTensor()

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            for sample in valid_data_list:
                inputFeaturesLabelsForProcess, valid_question_tokens = self.process_sample(sample, valid_tableData,
                                                                                           bert_tokenizer)
                ValidFeaturesLabels.sequence_labeling_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_inputMask)
                ValidFeaturesLabels.select_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.select_column_inputMask)
                ValidFeaturesLabels.where_conlumn_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.where_conlumn_inputMask)
                ValidFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                ValidFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                ValidFeaturesLabels.where_relation_labelList.extend(inputFeaturesLabelsForProcess.where_relation_label)
                ValidFeaturesLabels.select_agg_labelList.extend(inputFeaturesLabelsForProcess.select_agg_label)
                ValidFeaturesLabels.sequence_labeling_labelList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_label)
                ValidFeaturesLabels.where_column_number_labelList.extend(
                    inputFeaturesLabelsForProcess.where_column_number_label)
                ValidFeaturesLabels.sel_where_detemine_labellist.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_label)
                ValidFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                ValidFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                ValidFeaturesLabels.column_queryList.extend(inputFeaturesLabelsForProcess.column_queryList)
                ValidFeaturesLabels.column_tableidList.extend(inputFeaturesLabelsForProcess.column_tableidList)
                ValidFeaturesLabels.each_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.each_column_inputMask)
                ValidFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                ValidFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                ValidFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                ValidFeaturesLabels.select_number_labelList.extend(inputFeaturesLabelsForProcess.select_number_label)
                ValidFeaturesLabels.where_number_labelList.extend(inputFeaturesLabelsForProcess.where_number_label)
                ValidFeaturesLabels.where_op_labelList.extend(inputFeaturesLabelsForProcess.where_op_label)
                ValidFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question_tokens)
                ValidFeaturesLabels.eachData_indexList.append(len(ValidFeaturesLabels.connect_inputIDsList))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            for sample in test_data_list:
                inputFeaturesLabelsForProcess, test_question_tokens = self.process_sample_test(sample, test_tableData,
                                                                                               bert_tokenizer)
                TestFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                TestFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                TestFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                TestFeaturesLabels.each_column_inputMaskList.extend(inputFeaturesLabelsForProcess.each_column_inputMask)
                TestFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                TestFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                TestFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                TestFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                TestFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                TestFeaturesLabels.question_token_list.append(test_question_tokens)
                TestFeaturesLabels.eachData_indexList.append(len(TestFeaturesLabels.connect_inputIDsList))
                TestFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TestFeaturesLabels.table_id_list.append(sample["table_id"])

            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_conlumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_relation_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_agg_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_column_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_labellist,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.firstColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.each_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_op_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )
            test_dataset = data.TensorDataset(torch.tensor(TestFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.attention_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.firstColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.each_column_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.question_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.value_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.sel_where_detemine_inputMaskList,
                                                           dtype=torch.long),
                                              )

            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.step_batch_size, shuffle=False)
            test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.step_batch_size, shuffle=False)

            return valid_iterator, ValidFeaturesLabels, valid_tableData, test_iterator, TestFeaturesLabels, test_tableData
        else:
            print(
                "There is no such mode for data_iterator, please select mode from \"train, test, evaluate, test&evaluate\"")

    def data_iterator_(self, mode='train'):
        # train: 41522 val: 4396 test: 4086
        if mode == 'train':
            logger.info("Loading data, the mode is \"train\", Loading train_set and valid_set")
            train_queryData = self.read_query(self.train_data_path)
            train_tableData = self.read_table(self.train_table_path)
            valid_queryData = self.read_query(self.valid_data_path)
            valid_tableData = self.read_table(self.valid_table_path)
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
            TrainFeaturesLabels = InputFeaturesLabelsForTensor()
            ValidFeaturesLabels = InputFeaturesLabelsForTensor()

            for sample in train_queryData:
                inputFeaturesLabelsForProcess, train_question_tokens = self.process_sample(sample, train_tableData,
                                                                                           bert_tokenizer)
                TrainFeaturesLabels.sequence_labeling_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_inputMask)
                TrainFeaturesLabels.select_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.select_column_inputMask)
                TrainFeaturesLabels.where_conlumn_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.where_conlumn_inputMask)
                TrainFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                TrainFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                TrainFeaturesLabels.where_relation_labelList.extend(inputFeaturesLabelsForProcess.where_relation_label)
                TrainFeaturesLabels.select_agg_labelList.extend(inputFeaturesLabelsForProcess.select_agg_label)
                TrainFeaturesLabels.sequence_labeling_labelList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_label)
                TrainFeaturesLabels.where_column_number_labelList.extend(
                    inputFeaturesLabelsForProcess.where_column_number_label)
                TrainFeaturesLabels.sel_where_detemine_labellist.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_label)
                TrainFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                TrainFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                TrainFeaturesLabels.column_queryList.extend(inputFeaturesLabelsForProcess.column_queryList)
                TrainFeaturesLabels.column_tableidList.extend(inputFeaturesLabelsForProcess.column_tableidList)
                TrainFeaturesLabels.each_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.each_column_inputMask)
                TrainFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                TrainFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                TrainFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                TrainFeaturesLabels.select_number_labelList.extend(inputFeaturesLabelsForProcess.select_number_label)
                TrainFeaturesLabels.where_number_labelList.extend(inputFeaturesLabelsForProcess.where_number_label)
                TrainFeaturesLabels.where_op_labelList.extend(inputFeaturesLabelsForProcess.where_op_label)
                TrainFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                TrainFeaturesLabels.question_token_list.append(train_question_tokens)
                TrainFeaturesLabels.eachData_indexList.append(len(TrainFeaturesLabels.connect_inputIDsList))
                TrainFeaturesLabels.sql_list.append(sample["sql"])
                TrainFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TrainFeaturesLabels.table_id_list.append(sample["table_id"])

            for sample in valid_queryData:
                inputFeaturesLabelsForProcess, valid_question_tokens = self.process_sample(sample, valid_tableData,
                                                                                           bert_tokenizer)
                ValidFeaturesLabels.sequence_labeling_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_inputMask)
                ValidFeaturesLabels.select_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.select_column_inputMask)
                ValidFeaturesLabels.where_conlumn_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.where_conlumn_inputMask)
                ValidFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                ValidFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                ValidFeaturesLabels.where_relation_labelList.extend(inputFeaturesLabelsForProcess.where_relation_label)
                ValidFeaturesLabels.select_agg_labelList.extend(inputFeaturesLabelsForProcess.select_agg_label)
                ValidFeaturesLabels.sequence_labeling_labelList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_label)
                ValidFeaturesLabels.where_column_number_labelList.extend(
                    inputFeaturesLabelsForProcess.where_column_number_label)
                ValidFeaturesLabels.sel_where_detemine_labellist.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_label)
                ValidFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                ValidFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                ValidFeaturesLabels.column_queryList.extend(inputFeaturesLabelsForProcess.column_queryList)
                ValidFeaturesLabels.column_tableidList.extend(inputFeaturesLabelsForProcess.column_tableidList)
                ValidFeaturesLabels.each_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.each_column_inputMask)
                ValidFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                ValidFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                ValidFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                ValidFeaturesLabels.select_number_labelList.extend(inputFeaturesLabelsForProcess.select_number_label)
                ValidFeaturesLabels.where_number_labelList.extend(inputFeaturesLabelsForProcess.where_number_label)
                ValidFeaturesLabels.where_op_labelList.extend(inputFeaturesLabelsForProcess.where_op_label)
                ValidFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question_tokens)
                ValidFeaturesLabels.eachData_indexList.append(len(ValidFeaturesLabels.connect_inputIDsList))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            train_dataset = data.TensorDataset(
                torch.tensor(TrainFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.sequence_labeling_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.select_column_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.where_conlumn_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.sel_where_detemine_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.attention_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.where_relation_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.select_agg_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.sequence_labeling_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.where_column_number_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.sel_where_detemine_labellist, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.firstColumn_CLS_startPositionList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.each_column_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.question_masks, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.nextColumn_CLS_startPositionList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.nextColumn_inputMaskList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.select_number_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.where_number_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.where_op_labelList, dtype=torch.long),
                torch.tensor(TrainFeaturesLabels.value_masks, dtype=torch.long))
            valid_dataset = data.TensorDataset(
                torch.tensor(ValidFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.sequence_labeling_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.select_column_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.where_conlumn_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.sel_where_detemine_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.attention_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.where_relation_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.select_agg_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.sequence_labeling_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.where_column_number_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.sel_where_detemine_labellist, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.firstColumn_CLS_startPositionList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.each_column_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.nextColumn_CLS_startPositionList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.nextColumn_inputMaskList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.select_number_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.where_number_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.where_op_labelList, dtype=torch.long),
                torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long))

            # 将 dataset 转成 dataloader
            train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=self.step_batch_size, shuffle=True)
            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.step_batch_size, shuffle=False)
            return train_iterator, valid_iterator, ValidFeaturesLabels, valid_tableData
        elif mode == "test":
            logger.info("Loading data, the mode is \"test\", Loading test_set")
            test_data_list = self.read_query(self.test_data_path)
            test_tableData = self.read_table(self.test_table_path)
            TestFeaturesLabels = InputFeaturesLabelsForTensor()
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
            TestFeaturesLabels = InputFeaturesLabelsForTensor()

            for sample in test_data_list:
                inputFeaturesLabelsForProcess, test_question_tokens = self.process_sample_test(sample, test_tableData,
                                                                                               bert_tokenizer)
                TestFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                TestFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                TestFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                TestFeaturesLabels.each_column_inputMaskList.extend(inputFeaturesLabelsForProcess.each_column_inputMask)
                TestFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                TestFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                TestFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                TestFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                TestFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                TestFeaturesLabels.question_token_list.append(test_question_tokens)
                TestFeaturesLabels.eachData_indexList.append(len(TestFeaturesLabels.connect_inputIDsList))
                TestFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TestFeaturesLabels.table_id_list.append(sample["table_id"])

            test_dataset = data.TensorDataset(torch.tensor(TestFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.attention_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.firstColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.each_column_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.question_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.value_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.sel_where_detemine_inputMaskList,
                                                           dtype=torch.long),
                                              )

            test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.step_batch_size, shuffle=False)

            return test_iterator, TestFeaturesLabels, test_tableData
        elif mode == "evaluate":
            logger.info("Loading data, the mode is \"evaluate\", Loading valid_set")
            valid_data_list = self.read_query(self.valid_data_path)
            valid_tableData = self.read_table(self.valid_table_path)

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            ValidFeaturesLabels = InputFeaturesLabelsForTensor()

            for sample in valid_data_list:
                inputFeaturesLabelsForProcess, valid_question_tokens = self.process_sample(sample, valid_tableData,
                                                                                           bert_tokenizer)
                ValidFeaturesLabels.sequence_labeling_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_inputMask)
                ValidFeaturesLabels.select_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.select_column_inputMask)
                ValidFeaturesLabels.where_conlumn_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.where_conlumn_inputMask)
                ValidFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                ValidFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                ValidFeaturesLabels.where_relation_labelList.extend(inputFeaturesLabelsForProcess.where_relation_label)
                ValidFeaturesLabels.select_agg_labelList.extend(inputFeaturesLabelsForProcess.select_agg_label)
                ValidFeaturesLabels.sequence_labeling_labelList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_label)
                ValidFeaturesLabels.where_column_number_labelList.extend(
                    inputFeaturesLabelsForProcess.where_column_number_label)
                ValidFeaturesLabels.sel_where_detemine_labellist.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_label)
                ValidFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                ValidFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                ValidFeaturesLabels.column_queryList.extend(inputFeaturesLabelsForProcess.column_queryList)
                ValidFeaturesLabels.column_tableidList.extend(inputFeaturesLabelsForProcess.column_tableidList)
                ValidFeaturesLabels.each_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.each_column_inputMask)
                ValidFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                ValidFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                ValidFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                ValidFeaturesLabels.select_number_labelList.extend(inputFeaturesLabelsForProcess.select_number_label)
                ValidFeaturesLabels.where_number_labelList.extend(inputFeaturesLabelsForProcess.where_number_label)
                ValidFeaturesLabels.where_op_labelList.extend(inputFeaturesLabelsForProcess.where_op_label)
                ValidFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question_tokens)
                ValidFeaturesLabels.eachData_indexList.append(len(ValidFeaturesLabels.connect_inputIDsList))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_conlumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_relation_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_agg_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_column_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_labellist,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.firstColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.each_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_op_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )

            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.step_batch_size, shuffle=False)

            return valid_iterator, ValidFeaturesLabels, valid_tableData  # , test_iterator, TestFeaturesLabels, test_tableData
        elif mode == "test&evaluate":
            logger.info("Loading data, the mode is \"test&evaluate\", Loading test_set and valid_set")
            test_data_list = self.read_query(self.test_data_path)
            test_tableData = self.read_table(self.test_table_path)
            valid_data_list = self.read_query(self.valid_data_path)
            valid_tableData = self.read_table(self.valid_table_path)

            TestFeaturesLabels = InputFeaturesLabelsForTensor()
            ValidFeaturesLabels = InputFeaturesLabelsForTensor()

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            for sample in valid_data_list:
                inputFeaturesLabelsForProcess, valid_question_tokens = self.process_sample(sample, valid_tableData,
                                                                                           bert_tokenizer)
                ValidFeaturesLabels.sequence_labeling_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_inputMask)
                ValidFeaturesLabels.select_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.select_column_inputMask)
                ValidFeaturesLabels.where_conlumn_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.where_conlumn_inputMask)
                ValidFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                ValidFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                ValidFeaturesLabels.where_relation_labelList.extend(inputFeaturesLabelsForProcess.where_relation_label)
                ValidFeaturesLabels.select_agg_labelList.extend(inputFeaturesLabelsForProcess.select_agg_label)
                ValidFeaturesLabels.sequence_labeling_labelList.extend(
                    inputFeaturesLabelsForProcess.sequence_labeling_label)
                ValidFeaturesLabels.where_column_number_labelList.extend(
                    inputFeaturesLabelsForProcess.where_column_number_label)
                ValidFeaturesLabels.sel_where_detemine_labellist.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_label)
                ValidFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                ValidFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                ValidFeaturesLabels.column_queryList.extend(inputFeaturesLabelsForProcess.column_queryList)
                ValidFeaturesLabels.column_tableidList.extend(inputFeaturesLabelsForProcess.column_tableidList)
                ValidFeaturesLabels.each_column_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.each_column_inputMask)
                ValidFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                ValidFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                ValidFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                ValidFeaturesLabels.select_number_labelList.extend(inputFeaturesLabelsForProcess.select_number_label)
                ValidFeaturesLabels.where_number_labelList.extend(inputFeaturesLabelsForProcess.where_number_label)
                ValidFeaturesLabels.where_op_labelList.extend(inputFeaturesLabelsForProcess.where_op_label)
                ValidFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question_tokens)
                ValidFeaturesLabels.eachData_indexList.append(len(ValidFeaturesLabels.connect_inputIDsList))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            for sample in test_data_list:
                inputFeaturesLabelsForProcess, test_question_tokens = self.process_sample_test(sample, test_tableData,
                                                                                               bert_tokenizer)
                TestFeaturesLabels.attention_inputMaskList.extend(inputFeaturesLabelsForProcess.attention_inputMask)
                TestFeaturesLabels.firstColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.firstColumn_CLS_startPosition)
                TestFeaturesLabels.connect_inputIDsList.extend(inputFeaturesLabelsForProcess.connect_inputIDs)
                TestFeaturesLabels.each_column_inputMaskList.extend(inputFeaturesLabelsForProcess.each_column_inputMask)
                TestFeaturesLabels.question_masks.extend(inputFeaturesLabelsForProcess.question_masks)
                TestFeaturesLabels.nextColumn_CLS_startPositionList.extend(
                    inputFeaturesLabelsForProcess.nextColumn_CLS_startPosition)
                TestFeaturesLabels.nextColumn_inputMaskList.extend(inputFeaturesLabelsForProcess.nextColumn_inputMask)
                TestFeaturesLabels.value_masks.extend(inputFeaturesLabelsForProcess.value_masks)
                TestFeaturesLabels.sel_where_detemine_inputMaskList.extend(
                    inputFeaturesLabelsForProcess.sel_where_detemine_inputMask)
                TestFeaturesLabels.question_token_list.append(test_question_tokens)
                TestFeaturesLabels.eachData_indexList.append(len(TestFeaturesLabels.connect_inputIDsList))
                TestFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TestFeaturesLabels.table_id_list.append(sample["table_id"])

            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_conlumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_relation_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_agg_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sequence_labeling_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_column_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_where_detemine_labellist,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.firstColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.each_column_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_CLS_startPositionList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.nextColumn_inputMaskList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.select_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_number_labelList,
                                                            dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_op_labelList, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )
            test_dataset = data.TensorDataset(torch.tensor(TestFeaturesLabels.connect_inputIDsList, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.attention_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.firstColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.each_column_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.question_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_CLS_startPositionList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.nextColumn_inputMaskList,
                                                           dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.value_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.sel_where_detemine_inputMaskList,
                                                           dtype=torch.long),
                                              )

            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.step_batch_size, shuffle=False)
            test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.step_batch_size, shuffle=False)

            return valid_iterator, ValidFeaturesLabels, valid_tableData, test_iterator, TestFeaturesLabels, test_tableData
        else:
            print(
                "There is no such mode for data_iterator, please select mode from \"train, test, evaluate, test&evaluate\"")

    def test(self, batch_size, step_batch_size, do_evaluate=True, do_test=True):
        self.batch_size = batch_size
        self.step_batch_size = step_batch_size
        # print('load data')
        # train_iterator, valid_iterator, valid_question_list, valid_table_id_list, valid_eachData_index, valid_sql_list, valid_tableData, valid_column_queryList, valid_column_tableidList, test_iterator, test_question_list, test_table_id_list, test_eachData_index, test_tableData, valid_question_tokens, test_question_tokens = self.data_iterator()
        self.seed_everything()
        model = BertNL2SQL(self.bert_config)
        model.load_state_dict(torch.load(os.path.join(self.model_dir, "model2019-08-09.bin")))
        model = model.to(self.device) if torch.cuda.is_available() else model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        if do_evaluate:
            valid_iterator, ValidFeaturesLabels, valid_tableData = self.data_iterator(mode="evaluate")
            sequence_labeling_predict = []
            select_agg_predict = []
            where_relation_predict = []
            where_conlumn_number_predict = []
            sel_where_detemine_predict = []
            sequence_labeling_groudTruth = []
            select_agg_groundTruth = []
            where_relation_groundTruth = []
            where_conlumn_number_groundTruth = []
            sel_where_detemine_groundTruth = []
            firstColumn_CLS_startPositionList = []
            select_number_groundTruth = []
            where_number_groundTruth = []
            select_number_predict = []
            where_number_predict = []
            type_probs_list = []
            where_op_groundTruth = []
            where_op_predict = []
            for j, valid_batch_data in enumerate(valid_iterator):
                if torch.cuda.is_available():
                    print('validbatchIndex:', j)

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
                    nextColumn_CLS_startPositionList = valid_batch_data[14].to(self.device)
                    subheader_masks = valid_batch_data[15].to(self.device)
                    sel_num_labels = valid_batch_data[16].to(self.device)
                    where_num_labels = valid_batch_data[17].to(self.device)
                    op_labels = valid_batch_data[18].to(self.device)
                    value_masks = valid_batch_data[19].to(self.device)
                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(
                    input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                    nextColumn_CLS_startPositionList, value_masks, cls_indices)

                connection_labels = connection_labels.to('cpu').numpy().tolist()
                agg_labels = agg_labels.to('cpu').numpy().tolist()
                tag_labels = tag_labels.to('cpu').numpy().tolist()
                con_num_labels = con_num_labels.to('cpu').numpy().tolist()
                type_labels = type_labels.to('cpu').numpy().tolist()
                cls_indices = cls_indices.to('cpu').numpy().tolist()
                sel_num_labels = sel_num_labels.to('cpu').numpy().tolist()
                where_num_labels = where_num_labels.to('cpu').numpy().tolist()
                op_labels = op_labels.to('cpu').numpy().tolist()

                sequence_labeling_predict.extend(tag_logits)
                select_agg_predict.extend(agg_logits)
                where_relation_predict.extend(connection_logits)
                where_conlumn_number_predict.extend(con_num_logits)
                sel_where_detemine_predict.extend(type_logits)
                sequence_labeling_groudTruth.extend(tag_labels)
                select_agg_groundTruth.extend(agg_labels)
                where_relation_groundTruth.extend(connection_labels)
                where_conlumn_number_groundTruth.extend(con_num_labels)
                sel_where_detemine_groundTruth.extend(type_labels)
                firstColumn_CLS_startPositionList.extend(cls_indices)
                select_number_groundTruth.extend(sel_num_labels)
                where_number_groundTruth.extend(where_num_labels)
                select_number_predict.extend(sel_num_logits)
                where_number_predict.extend(where_num_logits)
                type_probs_list.extend(type_probs)
                where_op_groundTruth.extend(op_labels)
                where_op_predict.extend(op_logits)

            logits_lists = [sequence_labeling_predict, select_agg_predict, where_relation_predict,
                            where_conlumn_number_predict, sel_where_detemine_predict, select_number_predict,
                            where_number_predict, type_probs_list, where_op_predict]
            labels_lists = [sequence_labeling_groudTruth, select_agg_groundTruth, where_relation_groundTruth,
                            where_conlumn_number_groundTruth, sel_where_detemine_groundTruth, select_number_groundTruth,
                            where_number_groundTruth, where_op_groundTruth]
            logical_acc = Evaluate.evaluate(
                logits_lists, firstColumn_CLS_startPositionList, labels_lists,
                ValidFeaturesLabels.question_list, ValidFeaturesLabels.question_token_list,
                ValidFeaturesLabels.table_id_list, ValidFeaturesLabels.eachData_indexList,
                ValidFeaturesLabels.sql_list, valid_tableData,
                ValidFeaturesLabels.column_queryList, ValidFeaturesLabels.column_tableidList)
            logger.info("\nlogical_acc: %.3f\n\n" % (logical_acc))

        if do_test:
            test_iterator, TestFeaturesLabels, test_tableData = self.data_iterator(mode="test")
            print('Start predicting')
            sequence_labeling_predict = []
            select_agg_predict = []
            where_relation_predict = []
            where_conlumn_number_predict = []
            sel_where_detemine_predict = []
            firstColumn_CLS_startPositionList = []
            select_number_predict = []
            where_number_predict = []
            type_probs_list = []
            where_op_predict = []
            for j, test_batch_data in enumerate(test_iterator):
                if j%100==0:
                    print('testbatchIndex:', j)

                if torch.cuda.is_available():
                    input_ids = test_batch_data[0].to(self.device)
                    attention_masks = test_batch_data[1].to(self.device)
                    cls_indices = test_batch_data[2].to(self.device)
                    header_masks = test_batch_data[3].to(self.device)
                    question_masks = test_batch_data[4].to(self.device)
                    nextColumn_CLS_startPositionList = test_batch_data[5].to(self.device)
                    subheader_masks = test_batch_data[6].to(self.device)
                    value_masks = test_batch_data[7].to(self.device)
                    type_masks = test_batch_data[8].to(self.device)
                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(
                    input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                    nextColumn_CLS_startPositionList, value_masks, cls_indices)
                sequence_labeling_predict.extend(tag_logits)
                select_agg_predict.extend(agg_logits)
                where_relation_predict.extend(connection_logits)
                where_conlumn_number_predict.extend(con_num_logits)
                sel_where_detemine_predict.extend(type_logits)
                firstColumn_CLS_startPositionList.extend(cls_indices)
                select_number_predict.extend(sel_num_logits)
                where_number_predict.extend(where_num_logits)
                type_probs_list.extend(type_probs)
                where_op_predict.extend(op_logits)

            logits_lists = [sequence_labeling_predict, select_agg_predict, where_relation_predict,
                            where_conlumn_number_predict, sel_where_detemine_predict, select_number_predict,
                            where_number_predict, type_probs_list, where_op_predict]
            labels_lists = [[] for _ in range(8)]
            test_sql_list, test_column_queryList, test_column_tableidList = [], [], []
            Evaluate.evaluate(logits_lists, firstColumn_CLS_startPositionList, labels_lists,
                              TestFeaturesLabels.question_list, TestFeaturesLabels.question_token_list,
                              TestFeaturesLabels.table_id_list, TestFeaturesLabels.eachData_indexList, test_sql_list,
                              test_tableData,
                              test_column_queryList, test_column_tableidList, config, do_test=True)

    def train(self):
        if self.debug: self.epochs = 1
        # 加载 dataloader
        train_loader, valid_loader, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list, valid_question_token_list = self.data_iterator()
        # 训练
        self.seed_everything()
        lr = 1e-5
        accumulation_steps = math.ceil(self.batch_size / self.step_batch_size)
        # 加载预训练模型
        model = BertNL2SQL.from_pretrained(self.bert_model_path, cache_dir=None)
        model.zero_grad()
        if torch.cuda.is_available():
            model = model.to(self.device)
        # 不同的参数组设置不同的 weight_decay
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        epoch_steps = int(train_loader.sampler.num_samples / self.step_batch_size / accumulation_steps)
        num_train_optimization_steps = int(self.epochs * epoch_steps)
        valid_every = math.floor(epoch_steps * accumulation_steps / 5)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps)
        # 开始训练
        best_score = 0
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # 加载每个 batch 并训练
            for i, batch_data in enumerate(train_loader):
                if torch.cuda.is_available():
                    print('epoch:', epoch, 'batchIndex:', i)
                    input_ids = batch_data[0].to(self.device)
                    tag_masks = batch_data[1].to(self.device)
                    sel_masks = batch_data[2].to(self.device)
                    con_masks = batch_data[3].to(self.device)
                    type_masks = batch_data[4].to(self.device)
                    attention_masks = batch_data[5].to(self.device)
                    connection_labels = batch_data[6].to(self.device)
                    agg_labels = batch_data[7].to(self.device)
                    tag_labels = batch_data[8].to(self.device)
                    con_num_labels = batch_data[9].to(self.device)
                    type_labels = batch_data[10].to(self.device)
                    cls_index_list = batch_data[11].to(self.device)
                    header_masks = batch_data[12].to(self.device)
                    question_masks = batch_data[13].to(self.device)
                    subheader_cls_list = batch_data[14].to(self.device)
                    subheader_masks = batch_data[15].to(self.device)
                    sel_num_labels = batch_data[16].to(self.device)
                    where_num_labels = batch_data[17].to(self.device)
                    op_labels = batch_data[18].to(self.device)
                    value_masks = batch_data[19].to(self.device)
                else:
                    input_ids = batch_data[0]
                    tag_masks = batch_data[1]
                    sel_masks = batch_data[2]
                    con_masks = batch_data[3]
                    type_masks = batch_data[4]
                    attention_masks = batch_data[5]
                    connection_labels = batch_data[6]
                    agg_labels = batch_data[7]
                    tag_labels = batch_data[8]
                    con_num_labels = batch_data[9]
                    type_labels = batch_data[10]
                    cls_index_list = batch_data[11]
                    header_masks = batch_data[12]
                    question_masks = batch_data[13]
                    subheader_cls_list = batch_data[14]
                    subheader_masks = batch_data[15]
                    sel_num_labels = batch_data[16]
                    where_num_labels = batch_data[17]
                    op_labels = batch_data[18]
                    value_masks = batch_data[19]
                if torch.sum(sel_masks) == 0 or torch.sum(con_masks) == 0 or torch.sum(tag_masks) == 0: continue
                train_dependencies = [tag_masks, sel_masks, con_masks, connection_labels, agg_labels, tag_labels,
                                      con_num_labels, type_labels, sel_num_labels, where_num_labels, op_labels]
                loss = model(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                             subheader_cls_list, value_masks, cls_index_list, train_dependencies=train_dependencies)
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
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
            for j, valid_batch_data in enumerate(valid_loader):
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
                else:
                    input_ids = valid_batch_data[0]
                    tag_masks = valid_batch_data[1]
                    sel_masks = valid_batch_data[2]
                    con_masks = valid_batch_data[3]
                    type_masks = valid_batch_data[4]
                    attention_masks = valid_batch_data[5]
                    connection_labels = valid_batch_data[6]
                    agg_labels = valid_batch_data[7]
                    tag_labels = valid_batch_data[8]
                    con_num_labels = valid_batch_data[9]
                    type_labels = valid_batch_data[10]
                    cls_indices = valid_batch_data[11]
                    header_masks = valid_batch_data[12]
                    question_masks = valid_batch_data[13]
                    subheader_cls_list = valid_batch_data[14]
                    subheader_masks = valid_batch_data[15]
                    sel_num_labels = valid_batch_data[16]
                    where_num_labels = valid_batch_data[17]
                    op_labels = valid_batch_data[18]
                    value_masks = valid_batch_data[19]
                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(
                    input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                    subheader_cls_list, value_masks, cls_indices)

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

            logits_lists = [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list,
                            type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list,
                            op_logits_list]
            labels_lists = [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list,
                            type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list]
            logical_acc = Evaluate.evaluate(logits_lists, cls_index_list, labels_lists, valid_question_list,
                                            valid_question_token_list, valid_table_id_list, valid_sample_index_list,
                                            valid_sql_list, valid_table_dict, valid_header_question_list,
                                            valid_header_table_id_list)

            score = logical_acc
            # logger.info("\nlogical_acc: %.3f\n\n" % (logical_acc))
            logger.info("\nepoch: %d, logical_acc: %.3f\n\n" % (epoch, logical_acc))

            if not self.debug and score > best_score:
                best_score = score
                state_dict = model.state_dict()
                model_name = "../model/model{}.bin".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))
                torch.save(state_dict, model_name)
            model.train()
        # del 训练相关输入和模型
        training_history = [train_loader, valid_loader, model, optimizer, param_optimizer, optimizer_grouped_parameters]
        for variable in training_history:
            del variable
        gc.collect()

    def train_(self, batch_size, step_batch_size):
        self.batch_size = batch_size
        self.step_batch_size = step_batch_size
        if self.debug: self.epochs = 1
        with timer('加载数据'):
            train_iterator, valid_iterator, ValidFeaturesLabels, valid_tableData = self.data_iterator(mode="train")

        # 训练
        self.seed_everything()
        lr = 1e-5
        accumulation_steps = math.ceil(self.batch_size / self.step_batch_size)
        # 预训练 bert 转成 pytorch
        # 加载预训练模型
        model = BertNL2SQL.from_pretrained(self.bert_model_path, cache_dir=None)
        model.zero_grad()
        if torch.cuda.is_available():
            model = model.to(self.device)
        # 不同的参数组设置不同的 weight_decay
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        epoch_steps = int(train_iterator.sampler.num_samples / self.step_batch_size / accumulation_steps)
        num_train_optimization_steps = int(self.epochs * epoch_steps)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps)
        # 开始训练
        best_score = 0
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # 加载每个 batch 并训练
            for i, batch_data in enumerate(train_iterator):
                if torch.cuda.is_available():
                    print('epoch:', epoch, 'batchIndex:', i)
                    input_ids = batch_data[0].to(self.device)
                    tag_masks = batch_data[1].to(self.device)
                    sel_masks = batch_data[2].to(self.device)
                    con_masks = batch_data[3].to(self.device)
                    type_masks = batch_data[4].to(self.device)
                    attention_masks = batch_data[5].to(self.device)
                    connection_labels = batch_data[6].to(self.device)
                    agg_labels = batch_data[7].to(self.device)
                    tag_labels = batch_data[8].to(self.device)
                    con_num_labels = batch_data[9].to(self.device)
                    type_labels = batch_data[10].to(self.device)
                    firstColumn_CLS_startPositionList = batch_data[11].to(self.device)
                    header_masks = batch_data[12].to(self.device)
                    question_masks = batch_data[13].to(self.device)
                    nextColumn_CLS_startPositionList = batch_data[14].to(self.device)
                    subheader_masks = batch_data[15].to(self.device)
                    sel_num_labels = batch_data[16].to(self.device)
                    where_num_labels = batch_data[17].to(self.device)
                    op_labels = batch_data[18].to(self.device)
                    value_masks = batch_data[19].to(self.device)
                if torch.sum(sel_masks) == 0 or torch.sum(con_masks) == 0 or torch.sum(tag_masks) == 0: continue
                train_dependencies = [tag_masks, sel_masks, con_masks, connection_labels, agg_labels, tag_labels,
                                      con_num_labels, type_labels, sel_num_labels, where_num_labels, op_labels]
                loss = model(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                             nextColumn_CLS_startPositionList, value_masks, firstColumn_CLS_startPositionList,
                             train_dependencies=train_dependencies)
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            # 开始验证
            model.eval()
            sequence_labeling_predict = []
            select_agg_predict = []
            where_relation_predict = []
            where_conlumn_number_predict = []
            sel_where_detemine_predict = []
            sequence_labeling_groudTruth = []
            select_agg_groundTruth = []
            where_relation_groundTruth = []
            where_conlumn_number_groundTruth = []
            sel_where_detemine_groundTruth = []
            firstColumn_CLS_startPositionList = []
            select_number_groundTruth = []
            where_number_groundTruth = []
            select_number_predict = []
            where_number_predict = []
            type_probs_list = []
            where_op_groundTruth = []
            where_op_predict = []
            for j, valid_batch_data in enumerate(valid_iterator):
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
                    nextColumn_CLS_startPositionList = valid_batch_data[14].to(self.device)
                    subheader_masks = valid_batch_data[15].to(self.device)
                    sel_num_labels = valid_batch_data[16].to(self.device)
                    where_num_labels = valid_batch_data[17].to(self.device)
                    op_labels = valid_batch_data[18].to(self.device)
                    value_masks = valid_batch_data[19].to(self.device)
                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(
                    input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                    nextColumn_CLS_startPositionList, value_masks, cls_indices)

                connection_labels = connection_labels.to('cpu').numpy().tolist()
                agg_labels = agg_labels.to('cpu').numpy().tolist()
                tag_labels = tag_labels.to('cpu').numpy().tolist()
                con_num_labels = con_num_labels.to('cpu').numpy().tolist()
                type_labels = type_labels.to('cpu').numpy().tolist()
                cls_indices = cls_indices.to('cpu').numpy().tolist()
                sel_num_labels = sel_num_labels.to('cpu').numpy().tolist()
                where_num_labels = where_num_labels.to('cpu').numpy().tolist()
                op_labels = op_labels.to('cpu').numpy().tolist()

                sequence_labeling_predict.extend(tag_logits)
                select_agg_predict.extend(agg_logits)
                where_relation_predict.extend(connection_logits)
                where_conlumn_number_predict.extend(con_num_logits)
                sel_where_detemine_predict.extend(type_logits)
                sequence_labeling_groudTruth.extend(tag_labels)
                select_agg_groundTruth.extend(agg_labels)
                where_relation_groundTruth.extend(connection_labels)
                where_conlumn_number_groundTruth.extend(con_num_labels)
                sel_where_detemine_groundTruth.extend(type_labels)
                firstColumn_CLS_startPositionList.extend(cls_indices)
                select_number_groundTruth.extend(sel_num_labels)
                where_number_groundTruth.extend(where_num_labels)
                select_number_predict.extend(sel_num_logits)
                where_number_predict.extend(where_num_logits)
                type_probs_list.extend(type_probs)
                where_op_groundTruth.extend(op_labels)
                where_op_predict.extend(op_logits)

            logits_lists = [sequence_labeling_predict, select_agg_predict, where_relation_predict,
                            where_conlumn_number_predict, sel_where_detemine_predict, select_number_predict,
                            where_number_predict, type_probs_list, where_op_predict]
            labels_lists = [sequence_labeling_groudTruth, select_agg_groundTruth, where_relation_groundTruth,
                            where_conlumn_number_groundTruth, sel_where_detemine_groundTruth, select_number_groundTruth,
                            where_number_groundTruth, where_op_groundTruth]
            logical_acc = Evaluate.evaluate(logits_lists, firstColumn_CLS_startPositionList, labels_lists,
                                            ValidFeaturesLabels, valid_tableData)
            score = logical_acc
            # logger.info("\nlogical_acc: %.3f\n\n" % (logical_acc))
            logger.info("\nepoch: %d, logical_acc: %.3f\n\n" % (epoch, logical_acc))

            if not self.debug and score > best_score:
                best_score = score
                state_dict = model.state_dict()
                model_name = "../model/model{}.bin".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))
                torch.save(state_dict, model_name)

            model.train()
        # del 训练相关输入和模型
        training_history = [train_iterator, valid_iterator, model, optimizer, param_optimizer,
                            optimizer_grouped_parameters]
        for variable in training_history:
            del variable
        gc.collect()

def model_config():
    parser = argparse.ArgumentParser()
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data")
    data_arg.add_argument("--model_dir", type=str, default="./model")
    data_arg.add_argument("--log_dir", type=str, default="./code/log_bad_cases")
    data_arg.add_argument("--submit_dir", type=str, default="./submit")
    data_arg.add_argument("--log_path", type=str, default="./code/log_bad_cases/nl2sql.log")

    config = parser.parse_args()
    return config


if __name__ == "__main__":
    config = model_config()
    logger = get_train_logger(config.log_path)
    with timer('initializing'):
        data_dir = config.data_dir
        model_dir = config.model_dir
        nl2sql = NL2SQL(config, epochs=4, batch_size=16, step_batch_size=16, max_len=128, debug=False)
    # with timer('训练'):
    #     # nl2sql.train(batch_size=16, step_batch_size=16)
    #     nl2sql.train()
    # with timer('验证'):
    #     nl2sql.test(batch_size=64, step_batch_size=64, do_evaluate=True, do_test=False)
    with timer('predicting'):
        nl2sql.test(batch_size=32, step_batch_size=16, do_evaluate=False, do_test=True)
