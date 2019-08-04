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
from train import QuestionMatcher,ValueOptimizer,BertNeuralNet

class Predictor:
    def __init__(self, data_dir, model_name, epochs=1, batch_size=64, base_batch_size=32, max_len=120, part=1., seed=1234, debug_mode=False):
        self.device = torch.device('cuda')
        self.data_dir = data_dir
        self.debug_mode = debug_mode
        self.model_name = model_name
        self.seed = seed
        self.part = part
        self.seed_everything()
        self.max_len = max_len
        self.epochs = epochs
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.split_ratio = 0.80
        if os.path.exists(self.data_dir):
            self.train_data_path = os.path.join(self.data_dir, "train/train.json")
            self.train_table_path = os.path.join(self.data_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(self.data_dir, "val/val.json")
            self.valid_table_path = os.path.join(self.data_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(self.data_dir, "test/test.json")
            self.test_table_path = os.path.join(self.data_dir, "test/test.tables.json")
            self.bert_model_path = "./chinese_wwm_L-12_H-768_A-12/"
            self.pytorch_bert_path =  "./chinese_wwm_L-12_H-768_A-12/pytorch_model.bin"
            self.bert_config = BertConfig("./chinese_wwm_L-12_H-768_A-12/config.json")
        else:
            input_dir = "/home1/lsy2018/NL2SQL/XSQL/data"
            self.train_data_path = os.path.join(input_dir, "train/train.json")
            self.train_table_path = os.path.join(input_dir, "train/train.tables.json")
            self.valid_data_path = os.path.join(input_dir, "val/val.json")
            self.valid_table_path = os.path.join(input_dir, "val/val.tables.json")
            self.test_data_path = os.path.join(input_dir, "test/test.json")
            self.test_table_path = os.path.join(input_dir, "test/test.tables.json")
            self.bert_model_path = os.path.join('/home1/lsy2018/NL2SQL/python5', "chinese_wwm_L-12_H-768_A-12/")
            self.pytorch_bert_path = os.path.join('/home1/lsy2018/NL2SQL/python5', "/chinese_wwm_L-12_H-768_A-12/pytorch_model.bin")
            self.bert_config = BertConfig(os.path.join('/home1/lsy2018/NL2SQL/python5', "chinese_wwm_L-12_H-768_A-12/bert_config.json"))
    def load_data(self, path, num=None):
        data_list = []
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if self.debug_mode and i == 200: break
                sample = json.loads(line)
                data_list.append(sample)
        if num and not self.debug_mode:
            random.seed(self.seed)
            data_list = random.sample(data_list, num)
        print(len(data_list))
        return data_list

    def load_table(self, path):
        table_dict = {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                table = json.loads(line)
                table_dict[table["id"]] = table
        return table_dict

    def seed_everything(self):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def convert_lines(self, text_series, max_seq_length, bert_tokenizer):
        max_seq_length -= 2
        all_tokens = []
        for text in text_series:
            tokens = bert_tokenizer.tokenize(text)
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
            one_token = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]) + [0] * (max_seq_length - len(tokens))
            all_tokens.append(one_token)
        return np.array(all_tokens)

    def create_mask(self, max_len, start_index, mask_len):
        mask = [0] * max_len
        for i in range(start_index, start_index + mask_len):
            mask[i] = 1
        return mask


    def process_valid_sample(self, sample, table_dict, bert_tokenizer):
        question = sample["question"]
        table_id = sample["table_id"]
        sel_list = sample["sql"]["sel"]
        agg_list = sample["sql"]["agg"]
        con_list = sample["sql"]["conds"]
        connection = sample["sql"]["cond_conn_op"]
        table_title = table_dict[table_id]["title"]
        table_header_list = table_dict[table_id]["header"]
        table_row_list = table_dict[table_id]["rows"]

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
        duplicate_indices = QuestionMatcher.duplicate_relative_index(con_list)
        con_dict = {}
        for [con_col, op, value], duplicate_index in zip(con_list, duplicate_indices):  # duplicate index 是跟着 value 的
            value = value.strip()
            matched_value, matched_index = QuestionMatcher.match_value(question, value, duplicate_index)
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
        question_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"])
        header_cls_index = len(question_ids)
        question_mask = self.create_mask(max_len=self.max_len, start_index=1, mask_len=len(question_tokens))
        # tag_list = sample_tag_logits[j][1: cls_index - 1]
        for col in range(len(table_header_list)):
            header = table_header_list[col]
            value_set = col_dict[header]
            header_tokens = bert_tokenizer.tokenize(header)
            header_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + header_tokens + ["[SEP]"])
            header_mask = self.create_mask(max_len=self.max_len, start_index=len(question_ids) + 1, mask_len=len(header_tokens))

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
            subheader_mask_len = len(conc_ids) - subheader_start_index - 1
            subheader_mask = self.create_mask(max_len=self.max_len, start_index=subheader_start_index, mask_len=subheader_mask_len)
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
            value_mask_len = len(conc_ids) - value_start_index - 1
            value_mask = self.create_mask(max_len=self.max_len, start_index=value_start_index, mask_len=value_mask_len)
            attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(conc_ids))
            conc_ids = conc_ids + [0] * (self.max_len - len(conc_ids))

            # TODO: [4] 改成了 [0]
            tag_ids = [0] * len(conc_ids)  # 4 是不标注

            op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}

            sel_mask, con_mask, type_mask = 0, 0, 1
            connection_id, agg_id, con_num, op = 0, 0, 0, 2
            if col in con_dict:
                # 如果 header 对应多个 values，values 必须全部匹配上才进入训练
                if list(map(lambda x: x[0], con_list)).count(col) != len(con_dict[col]): continue
                header_con_list = con_dict[col]
                for [op, value, index] in header_con_list:
                    # TODO: [op] 改成了 [1]
                    tag_ids[index + 1: index + 1 + len(value)] = [1] * len(value)
                tag_mask = [0] + [1] * len(question) + [0] * (self.max_len - len(question) - 1)
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

        return tag_masks, sel_masks, con_masks, type_masks, attention_masks, connection_labels, agg_labels, tag_labels, con_num_labels, type_labels, cls_index_list, conc_tokens, header_question_list, header_table_id_list, header_masks, question_masks, subheader_cls_list, subheader_masks, sel_num_labels, where_num_labels, op_labels, value_masks

    def process_test_sample(self, sample, table_dict, bert_tokenizer):
        question = sample["question"]
        table_id = sample["table_id"]
        table_title = table_dict[table_id]["title"]
        table_header_list = table_dict[table_id]["header"]
        table_row_list = table_dict[table_id]["rows"]

        col_dict = {header_name: set() for header_name in table_header_list}
        for row in table_row_list:
            for col, value in enumerate(row):
                header_name = table_header_list[col]
                col_dict[header_name].add(str(value))
     
        #(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks, subheader_cls_list, value_masks, cls_indices)

        conc_tokens = []
        attention_masks = []
        cls_index_list = []
        header_masks = []
        question_masks = []
        subheader_masks = []
        subheader_cls_list = []
        value_masks = []
        header_question_list = []
        header_table_id_list = []
        type_masks = []




        question_tokens = bert_tokenizer.tokenize(question)
        question_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + question_tokens + ["[SEP]"])
        header_cls_index = len(question_ids)
        question_mask = self.create_mask(max_len=self.max_len, start_index=1, mask_len=len(question_tokens))

        for col in range(len(table_header_list)):
            header = table_header_list[col]
            value_set = col_dict[header]
            header_tokens = bert_tokenizer.tokenize(header)
            header_ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + header_tokens + ["[SEP]"])
            header_mask = self.create_mask(max_len=self.max_len, start_index=len(question_ids) + 1, mask_len=len(header_tokens))

            conc_ids = question_ids + header_ids
            type_mask = 1

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
            subheader_mask_len = len(conc_ids) - subheader_start_index - 1
            subheader_mask = self.create_mask(max_len=self.max_len, start_index=subheader_start_index, mask_len=subheader_mask_len)
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
            value_mask_len = len(conc_ids) - value_start_index - 1
            value_mask = self.create_mask(max_len=self.max_len, start_index=value_start_index, mask_len=value_mask_len)
            attention_mask = self.create_mask(max_len=self.max_len, start_index=0, mask_len=len(conc_ids))
            conc_ids = conc_ids + [0] * (self.max_len - len(conc_ids))

            # TODO: [4] 改成了 [0]
            tag_ids = [0] * len(conc_ids)  # 4 是不标注

            op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}


            conc_tokens.append(conc_ids)
            attention_masks.append(attention_mask)
            cls_index_list.append(header_cls_index)
            header_question_list.append(question)
            header_table_id_list.append(table_id)
            header_masks.append(header_mask)
            question_masks.append(question_mask)
            subheader_cls_list.append(subheader_cls_index)
            subheader_masks.append(subheader_mask)
            value_masks.append(value_mask)
            type_masks.append(type_mask)


        return type_masks,attention_masks,header_masks,question_masks,subheader_masks,subheader_cls_list, value_masks,conc_tokens,cls_index_list, header_question_list, header_table_id_list,  



    def create_dataloader(self):
        valid_data_list = self.load_data(self.valid_data_path)
        valid_table_dict = self.load_table(self.valid_table_path)
        test_data_list = self.load_data(self.test_data_path)
        test_table_dict = self.load_table(self.test_table_path)

        bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

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

        for sample in valid_data_list:
            processed_result = self.process_valid_sample(sample, valid_table_dict, bert_tokenizer)
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
            valid_sample_index_list.append(len(valid_conc_tokens))
            valid_sql_list.append(sample["sql"])
            valid_question_list.append(sample["question"])
            valid_table_id_list.append(sample["table_id"])


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

        # valid_tag_masks.extend(processed_result[0])
        # valid_con_masks.extend(processed_result[2])
        # valid_type_masks.extend(processed_result[3])
        # valid_attention_masks.extend(processed_result[4])
        # valid_cls_index_list.extend(processed_result[10])
        # valid_conc_tokens.extend(processed_result[11])
        # valid_header_question_list.extend(processed_result[12])
        # valid_header_table_id_list.extend(processed_result[13])
        # valid_header_masks.extend(processed_result[14])
        # valid_question_masks.extend(processed_result[15])
        # valid_subheader_cls_list.extend(processed_result[16])
        # valid_subheader_masks.extend(processed_result[17])
        # valid_value_masks.extend(processed_result[21])
        # valid_sample_index_list.append(len(valid_conc_tokens))
        # valid_question_list.append(sample["question"])
        # valid_table_id_list.append(sample["table_id"])


        test_type_masks = []
        test_attention_masks = []
        test_header_masks = []
        test_question_masks = []
        test_subheader_masks = []
        test_subheader_cls_list = []
        test_value_masks = []
        test_question_list = []
        test_table_id_list = []
        test_conc_tokens = []
        test_sample_index_list = []
        test_cls_index_list = []
        test_header_question_list =[]
        test_header_table_id_list = []
        # test
        # type_masks,attention_masks, cls_index_list, conc_tokens, header_question_list, header_table_id_list, header_masks, question_masks, subheader_cls_list, subheader_masks, value_masks

        #(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks, subheader_cls_list, value_masks, cls_indices)

        for sample in test_data_list:
            processed_result = self.process_test_sample(sample, test_table_dict, bert_tokenizer)
            test_type_masks.extend(processed_result[0])
            test_attention_masks.extend(processed_result[1])
            test_header_masks.extend(processed_result[2])
            test_question_masks.extend(processed_result[3])
            test_subheader_masks.extend(processed_result[4])
            test_subheader_cls_list.extend(processed_result[5])
            test_value_masks.extend(processed_result[6])
            test_conc_tokens.extend(processed_result[7])
            test_sample_index_list.append(len(test_conc_tokens))
            test_cls_index_list.extend(processed_result[8])
            test_header_question_list.extend(processed_result[9])
            test_header_table_id_list.extend(processed_result[10])


            test_question_list.append(sample["question"])
            test_table_id_list.append(sample["table_id"])

        test_dataset = data.TensorDataset(
                                            torch.tensor(test_type_masks, dtype=torch.long),
                                            torch.tensor(test_attention_masks, dtype=torch.long),
                                            torch.tensor(test_header_masks,dtype = torch.long),
                                            torch.tensor(test_question_masks,dtype = torch.long),
                                            torch.tensor(test_subheader_masks,dtype = torch.long),
                                            torch.tensor(test_subheader_cls_list,dtype = torch.long),
                                            torch.tensor(test_value_masks,dtype = torch.long),
                                            torch.tensor(test_conc_tokens,dtype = torch.long),
                                            torch.tensor(test_cls_index_list, dtype=torch.long),
                                            )
        # 将 dataset 转成 dataloader
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.base_batch_size, shuffle=False)
        # 返回训练数据

        return valid_loader, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list,test_loader, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict,test_header_question_list,test_header_table_id_list

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def detail_score(self, y_true, y_pred, num_labels, ignore_num=None):
        detail_y_true = [[] for _ in range(num_labels)]
        detail_y_pred = [[] for _ in range(num_labels)]
        for i in range(len(y_pred)):
            for label in range(num_labels):
                if y_true[i] == label:
                    detail_y_true[label].append(1)
                else:
                    detail_y_true[label].append(0)
                if y_pred[i] == label:
                    detail_y_pred[label].append(1)
                else:
                    detail_y_pred[label].append(0)
        pre_list = []
        rec_list = []
        f1_list = []
        detail_output_str = ""
        for label in range(num_labels):
            if label == ignore_num: continue
            pre = precision_score(detail_y_true[label], detail_y_pred[label])
            rec = recall_score(detail_y_true[label], detail_y_pred[label])
            f1 = f1_score(detail_y_true[label], detail_y_pred[label])
            detail_output_str += "[%d] pre:%.3f rec:%.3f f1:%.3f\n" % (label, pre, rec, f1)
            pre_list.append(pre)
            rec_list.append(rec)
            f1_list.append(f1)
        acc = accuracy_score(y_true, y_pred)
        output_str = "overall_acc:%.3f, avg_pre:%.3f, avg_rec:%.3f, avg_f1:%.3f \n" % (acc, np.mean(pre_list), np.mean(rec_list), np.mean(f1_list))
        output_str += detail_output_str
        return output_str

    def sql_match(self, s1, s2):
        return (s1['cond_conn_op'] == s2['cond_conn_op']) & \
               (set(zip(s1['sel'], s1['agg'])) == set(zip(s2['sel'], s2['agg']))) & \
               (set([tuple(i) for i in s1['conds']]) == set([tuple(i) for i in s2['conds']]))


    def evaluate(self, logits_lists, cls_index_list, labels_lists, question_list, table_id_list, sample_index_list, correct_sql_list, table_dict, header_question_list, header_table_id_list):
        [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list, op_logits_list] = logits_lists
        [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list, type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list] = labels_lists

        # {"agg": [0], "cond_conn_op": 2, "sel": [1], "conds": [[3, 0, "11"], [6, 0, "11"]]}
        sql_dict = {"agg": [], "cond_conn_op": None, "sel": [], "conds": []}
        sql_list = []
        matched_num = 0
        for i in range(len(sample_index_list)):
            start_index = 0 if i == 0 else sample_index_list[i - 1]
            end_index = sample_index_list[i]
            sample_question = question_list[i]
            sample_table_id = table_id_list[i]
            sample_sql = correct_sql_list[i]
            sample_tag_logits = tag_logits_list[start_index: end_index]
            sample_agg_logits = agg_logits_list[start_index: end_index]
            sample_connection_logits = connection_logits_list[start_index: end_index]
            sample_con_num_logits = con_num_logits_list[start_index: end_index]
            sample_type_logits = type_logits_list[start_index: end_index]
            sample_sel_num_logits = sel_num_logits_list[start_index: end_index]
            sample_where_num_logits = where_num_logits_list[start_index: end_index]
            sample_op_logits_list = op_logits_list[start_index: end_index]

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
            for j, col_type in enumerate(sample_type_logits):
                type_probs = type_probs_list[j]
                sel_prob = type_probs[0]
                where_prob = type_probs[1]

                # sel
                agg = sample_agg_logits[j]
                sel_col = j
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
                for i in range(0, len(tag_list)):
                    a = len(tag_list)
                    b = len(sample_question)
                    current_tag = tag_list[i]
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
                        candidate_list[candidate_list_index][1].append(tag_list[i])
                    previous_tag = current_tag
                con_list = []
                # for candidate in candidate_list:
                for i in range(len(value_start_index_list)):
                    candidate = candidate_list[i]
                    value_start_index = value_start_index_list[i]
                    str_list = candidate[0]
                    if len(str_list) == 0: continue
                    value_str = "".join(str_list)
                    # print(sample_type_logits)
                    # print(table_header_list,value_str,sample_question,j)
                    # exit()
                    header = table_header_list[j]
                    col_data_type = table_type_list[j]
                    col_values = col_dict[j]
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
            sel_num = max(sample_sel_num_logits, key=sample_sel_num_logits.count)
            where_num = max(sample_where_num_logits, key=sample_where_num_logits.count)

            # connection = max(real_connection_list, key=real_connection_list.count) if where_num > 1 and len(real_connection_list) > 0 else 0
            # type_dict = {0: "sel", 1: "con", 2: "none"}
            sel_prob_list = sorted(sel_prob_list, key=lambda x: (-x["type"], x["prob"]), reverse=True)
            where_prob_list = sorted(where_prob_list, key=lambda x: (-(x["type"] ** 2 - 1) ** 2, x["prob"]), reverse=True)

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
            for j in range(min(where_num, len(where_prob_list))):
                tmp_sql_dict["conds"].append(where_prob_list[j]["cond"])
            sql_list.append(tmp_sql_dict)
            if self.sql_match(tmp_sql_dict, sample_sql):
                matched_num += 1
            """
            print(tmp_sql_dict)
            print(sample_sql)
            print(value_change_list)
            print("")
            """
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

    
    def generate_result(self,logits_lists,cls_index_list,question_list,table_id_list,sample_index_list,table_dict,header_question_list,header_table_id_list):

        [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list, op_logits_list] = logits_lists
        # {"agg": [0], "cond_conn_op": 2, "sel": [1], "conds": [[3, 0, "11"], [6, 0, "11"]]}
        sql_dict = {"agg": [], "cond_conn_op": None, "sel": [], "conds": []}
        sql_list = []
        for i in range(len(sample_index_list)):
            start_index = 0 if i == 0 else sample_index_list[i - 1]
            end_index = sample_index_list[i]
            sample_question = question_list[i]
            sample_table_id = table_id_list[i]
            sample_tag_logits = tag_logits_list[start_index: end_index]
            sample_agg_logits = agg_logits_list[start_index: end_index]
            sample_connection_logits = connection_logits_list[start_index: end_index]
            sample_con_num_logits = con_num_logits_list[start_index: end_index]
            sample_type_logits = type_logits_list[start_index: end_index]
            sample_sel_num_logits = sel_num_logits_list[start_index: end_index]
            sample_where_num_logits = where_num_logits_list[start_index: end_index]
            sample_op_logits_list = op_logits_list[start_index: end_index]

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
            for j, col_type in enumerate(sample_type_logits):
                type_probs = type_probs_list[j]
                sel_prob = type_probs[0]
                where_prob = type_probs[1]

                # sel
                agg = sample_agg_logits[j]
                sel_col = j
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
                for i in range(0, len(tag_list)):
                    a = len(tag_list)
                    b = len(sample_question)
                    current_tag = tag_list[i]
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
                        candidate_list[candidate_list_index][1].append(tag_list[i])
                    previous_tag = current_tag
                con_list = []
                # for candidate in candidate_list:
                for i in range(len(value_start_index_list)):
                    candidate = candidate_list[i]
                    value_start_index = value_start_index_list[i]
                    str_list = candidate[0]
                    if len(str_list) == 0: continue
                    value_str = "".join(str_list)
                    print(sample_type_logits)
                    print(table_header_list,value_str,sample_question,j)
                    header = table_header_list[j]
                    col_data_type = table_type_list[j]
                    col_values = col_dict[j]
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
            sel_num = max(sample_sel_num_logits, key=sample_sel_num_logits.count)
            where_num = max(sample_where_num_logits, key=sample_where_num_logits.count)

            # connection = max(real_connection_list, key=real_connection_list.count) if where_num > 1 and len(real_connection_list) > 0 else 0
            # type_dict = {0: "sel", 1: "con", 2: "none"}
            sel_prob_list = sorted(sel_prob_list, key=lambda x: (-x["type"], x["prob"]), reverse=True)
            where_prob_list = sorted(where_prob_list, key=lambda x: (-(x["type"] ** 2 - 1) ** 2, x["prob"]), reverse=True)

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
            for j in range(min(where_num, len(where_prob_list))):
                tmp_sql_dict["conds"].append(where_prob_list[j]["cond"])
            sql_list.append(tmp_sql_dict)
        fw = open("result.json", 'w')
        for sql_dict in sql_list:
            sql_dict_json = json.dumps(sql_dict, ensure_ascii=False)
            fw.write(sql_dict_json + '\n')
        fw.close()

    
    def inference(self, do_evaluate=True, do_test=False):
        if self.debug_mode: self.epochs = 1
        print('加载 dataloader')
        valid_loader, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list,test_loader, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict,test_header_question_list,test_header_table_id_list= self.create_dataloader()
        self.seed_everything()
        model = BertNeuralNet(self.bert_config)
        model.load_state_dict(torch.load("./my_model.bin"))
        model = model.to(self.device) if torch.cuda.is_available() else model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        if do_evaluate:
            print('开始验证')
            
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
                print('batchIndex:',j)

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

                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks, subheader_cls_list, value_masks, cls_indices)
                
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

            logits_lists = [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list, op_logits_list]
            labels_lists = [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list, type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list]
            eval_result, tag_acc, logical_acc = self.evaluate(logits_lists, cls_index_list, labels_lists, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list)
            score = logical_acc
            print('tag_acc:',tag_acc, 'logical_acc:',logical_acc)
            print(eval_result)

        if do_test:
            print('开始生成测试结果')
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

            for j,test_batch_data in enumerate(test_loader):
                print('batchIndex',j)

                # torch.tensor(test_type_masks, dtype=torch.long),
                # torch.tensor(test_attention_masks, dtype=torch.long),
                # torch.tensor(test_header_masks,dtype = torch.long),
                # torch.tensor(test_question_masks,dtype = torch.long),
                # torch.tensor(test_subheader_masks,dtype = torch.long),
                # torch.tensor(test_subheader_cls_list,dtype = torch.long),
                # torch.tensor(test_value_masks,dtype = torch.long),
                # torch.tensor(test_conc_tokens,dtype = torch.long),
                # torch.tensor(test_cls_index_list, dtype=torch.long),

        # test_dataset = data.TensorDataset(
        #                                     torch.tensor(test_type_masks, dtype=torch.long),
        #                                     torch.tensor(test_attention_masks, dtype=torch.long),
        #                                     torch.tensor(test_header_masks,dtype = torch.long),
        #                                     torch.tensor(test_question_masks,dtype = torch.long),
        #                                     torch.tensor(test_subheader_masks,dtype = torch.long),
        #                                     torch.tensor(test_subheader_cls_list,dtype = torch.long),
        #                                     torch.tensor(test_value_masks,dtype = torch.long),
        #                                     torch.tensor(test_conc_tokens,dtype = torch.long),
        #                                     torch.tensor(test_cls_index_list, dtype=torch.long),
        #                                     )


                type_masks = test_batch_data[0].to(self.device)
                attention_masks = test_batch_data[1].to(self.device)
                header_masks = test_batch_data[2].to(self.device)
                question_masks = test_batch_data[3].to(self.device)
                subheader_masks = test_batch_data[4].to(self.device)
                subheader_cls_list = test_batch_data[5].to(self.device)
                value_masks = test_batch_data[6].to(self.device)
                input_ids = test_batch_data[7].to(self.device)
                cls_indices = test_batch_data[8].to(self.device)



                # input_ids = test_batch_data[0].to(self.device)
                # tag_masks = test_batch_data[1].to(self.device)
                # sel_masks = test_batch_data[2].to(self.device)
                # con_masks = test_batch_data[3].to(self.device)
                # type_masks = test_batch_data[4].to(self.device)
                # attention_masks = test_batch_data[5].to(self.device)
                # connection_labels = test_batch_data[6].to(self.device)
                # agg_labels = test_batch_data[7].to(self.device)
                # tag_labels = test_batch_data[8].to(self.device)
                # con_num_labels = test_batch_data[9].to(self.device)
                # type_labels = test_batch_data[10].to(self.device)
                # cls_indices = test_batch_data[11].to(self.device)
                # header_masks = test_batch_data[12].to(self.device)
                # question_masks = test_batch_data[13].to(self.device)
                # subheader_cls_list = test_batch_data[14].to(self.device)
                # subheader_masks = test_batch_data[15].to(self.device)
                # sel_num_labels = test_batch_data[16].to(self.device)
                # where_num_labels = test_batch_data[17].to(self.device)
                # op_labels = test_batch_data[18].to(self.device)
                # value_masks = test_batch_data[19].to(self.device)


                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks, subheader_cls_list, value_masks, cls_indices)

                cls_indices = cls_indices.to('cpu').numpy().tolist()
                tag_logits_list.extend(tag_logits)
                agg_logits_list.extend(agg_logits)
                connection_logits_list.extend(connection_logits)
                con_num_logits_list.extend(con_num_logits)
                type_logits_list.extend(type_logits)
                sel_num_logits_list.extend(sel_num_logits)
                where_num_logits_list.extend(where_num_logits)
                type_probs_list.extend(type_probs)
                op_logits_list.extend(op_logits)
                cls_index_list.extend(cls_indices)
                        
            logits_lists = [tag_logits_list, agg_logits_list, connection_logits_list, con_num_logits_list, type_logits_list, sel_num_logits_list, where_num_logits_list, type_probs_list, op_logits_list]
            print('预测最终语句')
            
            
            #return test_loader, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict,test_header_question_list,test_header_table_id_list

            self.generate_result(logits_lists, cls_index_list, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict, test_header_question_list, test_header_table_id_list)





if __name__ == "__main__":
    data_dir = "./data"
    predictor = Predictor(data_dir, "model_name", epochs=6, batch_size=256, base_batch_size=32, max_len=120, part=1, debug_mode=False)
    time1 = time.time()
    predictor.inference(do_evaluate=False, do_test=True)
    print(time.time() - time1) 