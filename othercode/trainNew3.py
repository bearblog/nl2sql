#coding=utf-8

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
from Evaluate import Evaluate


class InputFeaturesLabels():
    def __init__(self):
        self.conc_tokens = []
        self.tag_masks = []
        self.sel_masks = []
        self.con_masks = []
        self.type_masks = []
        self.attention_masks = []
        self.connection_labels = []
        self.agg_labels = []
        self.tag_labels = []
        self.con_num_labels = []
        self.type_labels = []
        self.cls_index_list = []
        self.question_list = []
        self.table_id_list = []
        self.sample_index_list = []
        self.sql_list = []
        self.header_question_list = []
        self.header_table_id_list = []
        self.header_masks = []
        self.question_masks = []
        self.subheader_cls_list = []
        self.subheader_masks = []
        self.sel_num_labels = []
        self.where_num_labels = []
        self.op_labels = []
        self.value_masks = []
        self.question_token_list = []

class MatrixAttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MatrixAttentionLayer, self).__init__()
        self.linear_layer = nn.Linear(input_size, hidden_size, bias=False)

    def mask_and_acv(self, output, output_mask):
        # 对 output mask，然后求 Relu(WX)
        batch_size, output_len, hidden_size = output.shape
        output_mask = output_mask.unsqueeze(2).repeat(1, 1, hidden_size).view(batch_size, output_len, hidden_size)
        output = torch.mul(output, output_mask)
        output = F.relu(self.linear_layer(output))
        return output

    def forward(self, seq_output, seq_mask, target_output, target_mask):
        batch_size, seq_len, hidden_size = seq_output.shape
        batch_size, target_len, hidden_size = target_output.shape
        seq_mask = seq_mask.float()
        target_mask = target_mask.float()
        # 对 seq_output 和 target_output 都 mask，并 Relu
        seq_output_transformed = self.mask_and_acv(seq_output, seq_mask)
        target_output_transformed = self.mask_and_acv(target_output, target_mask)
        # seq_output_transformed 和 target_output_transformed 每一列两两点乘（处理成矩阵相乘），最后一个维度扩展成 hidden_size，最后是 (batch_size, seq_len, target_len, hidden_size)
        attention_matrix = torch.matmul(seq_output_transformed, target_output_transformed.transpose(2, 1))
        attention_matrix = F.softmax(attention_matrix.float(), dim=2)
        attention_matrix_unsqueeze = attention_matrix.unsqueeze(3).repeat(1, 1, 1, hidden_size)
        # 将 target_output 第二个维度 repeat seq_len，(batch_size, target_len, hidden_size) -> (batch_size, seq_len, target_len, hidden_size)，和注意力矩阵相乘后对第三个维度求sum
        target_output_unsqueeze = target_output.unsqueeze(1).repeat(1, seq_len, 1, 1)
        attention_output_unsqueeze = torch.mul(target_output_unsqueeze, attention_matrix_unsqueeze)
        attention_output = torch.sum(attention_output_unsqueeze, 2)
        # attention_output 拼接 seq_output，再用 seq_mask mask 一遍
        cat_output = torch.cat([seq_output, attention_output], 2)
        cat_mask = seq_mask.unsqueeze(2).repeat(1, 1, hidden_size * 2).view(batch_size, seq_len, hidden_size * 2)
        cat_output = torch.mul(cat_output, cat_mask)
        return cat_output, attention_output

class ColAttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ColAttentionLayer, self).__init__()
        self.linear_layer = nn.Linear(input_size, hidden_size, bias=False)

    def mask_and_acv(self, output, output_mask):
        # 对 output mask，然后求 Relu(WX)
        batch_size, output_len, hidden_size = output.shape
        output_mask = output_mask.unsqueeze(2).repeat(1, 1, hidden_size).view(batch_size, output_len, hidden_size)
        output = torch.mul(output, output_mask)
        output = F.relu(self.linear_layer(output))
        return output

    def forward(self, col_output, target_output, target_mask):
        batch_size, target_len, hidden_size = target_output.shape
        target_mask = target_mask.float()
        col_output_transformed = F.relu(self.linear_layer(col_output))
        target_output_transformed = self.mask_and_acv(target_output, target_mask)
        attention_matrix = torch.matmul(col_output_transformed.unsqueeze(1),
                                        target_output_transformed.transpose(2, 1)).squeeze(1)
        attention_matrix = F.softmax(attention_matrix.float(), dim=1)
        attention_matrix_unsqueeze = attention_matrix.unsqueeze(2).repeat(1, 1, hidden_size)
        attention_output_unsqueeze = torch.mul(target_output, attention_matrix_unsqueeze)
        attention_output = torch.sum(attention_output_unsqueeze, 1)
        cat_output = torch.cat([col_output, attention_output], 1)
        return cat_output, attention_output

class ValueOptimizer:
    @staticmethod
    def num_completion(value, question, start_index, end_index, value_start_index, value_end_index, num_type):
        num_set = set("0123456789") if num_type == "数字" else set("一二三四五六七八九十百千万亿")
        dot_str = "." if num_type == "数字" else "点"
        negative_str = "-" if num_type == "数字" else "负"
        pre_num = ""
        post_num = ""
        if start_index == 0 and value_start_index > 0:
            j = value_start_index - 1
            for j in range(value_start_index - 1, -2, -1):
                if j == -1:
                    break
                if question[j] == dot_str:
                    if j - 1 < 0 or question[j - 1] not in num_set:
                        break
                    else:
                        continue
                if question[j] == negative_str:
                    j -= 1
                    break
                if question[j] not in num_set:
                    break
            pre_num = question[j + 1: value_start_index]
        if end_index == len(value) and value_end_index < len(question) - 1:
            j = value_end_index + 1
            for j in range(value_end_index + 1, len(question) + 1):
                if j == len(question):
                    break
                if question[j] == dot_str:
                    if j + 1 >= len(question) or question[j + 1] not in num_set:
                        break
                    else:
                        continue
                if question[j] not in num_set:
                    break
            post_num = question[value_end_index + 1: j]
        return pre_num, post_num

    @staticmethod
    def find_longest_num(value, question, value_start_index):
        value = str(value)
        value_end_index = value_start_index + len(value) - 1
        longest_digit_num = None
        longest_chinese_num = None
        new_value = copy.copy(value)
        for i in range(len(value), 0, -1):
            is_match = re.search("[0-9.]{%d}" % i, value)
            if is_match:
                start_index = is_match.regs[0][0]
                end_index = is_match.regs[0][1]  # 最后一个index+1
                if start_index - 1 >= 0 and value[start_index - 1] == "-":
                    start_index -= 1
                longest_num = value[start_index: end_index]
                pre_num, post_num = ValueOptimizer.num_completion(value, question, start_index, end_index,
                                                                  value_start_index, value_end_index, num_type="数字")
                longest_digit_num = pre_num + longest_num + post_num
                new_value = pre_num + new_value + post_num
                break
        for i in range(len(value), 0, -1):
            value = value.replace("两百", "二百").replace("两千", "二百").replace("两万", "二百").replace("两亿", "二百")
            is_match = re.search("[点一二三四五六七八九十百千万亿]{%d}" % i, value)
            if is_match:
                start_index = is_match.regs[0][0]
                end_index = is_match.regs[0][1]
                if start_index - 1 >= 0 and value[start_index - 1] == "负":
                    start_index -= 1
                longest_num = value[start_index: end_index]
                pre_num, post_num = ValueOptimizer.num_completion(value, question, start_index, end_index,
                                                                  value_start_index, value_end_index, num_type="中文")
                longest_chinese_num = pre_num + longest_num + post_num
                new_value = pre_num + new_value + post_num
                break
        return new_value, longest_digit_num, longest_chinese_num

    @staticmethod
    def select_best_matched_value(value, col_values):
        value_char_dict = {}
        for char in value:
            if char in value_char_dict:
                value_char_dict[char] += 1
            else:
                value_char_dict[char] = 1
        col_values = set(col_values)
        max_matched_num = 0
        best_value = ""
        best_value_len = 100
        for col_value in col_values:
            char_dict = copy.copy(value_char_dict)
            matched_num = 0
            for char in col_value:
                if char in char_dict and char_dict[char] > 0:
                    matched_num += 1
                    char_dict[char] -= 1
            precision = matched_num / len(value)
            recall = matched_num / len(col_value)
            if matched_num > max_matched_num:
                max_matched_num = matched_num
                best_value = col_value
                best_value_len = len(col_value)
            elif matched_num > 0 and matched_num == max_matched_num and len(col_value) < best_value_len:
                best_value = col_value
                best_value_len = len(col_value)
        return best_value, max_matched_num

    @staticmethod
    def select_best_matched_value_from_candidates(candidate_values, col_values):
        max_matched_num = 0
        best_value = ""
        best_value_len = 100
        for value in candidate_values:
            value, matched_num = ValueOptimizer.select_best_matched_value(value, col_values)
            if matched_num > max_matched_num:
                max_matched_num = matched_num
                best_value = value
                best_value_len = len(value)
            elif matched_num == max_matched_num and len(value) < best_value_len:
                best_value = value
                best_value_len = len(value)
        return best_value

    @staticmethod
    def _chinese2digits(uchars_chinese):
        chinese_num_dict = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
        total = 0
        r = 1  # 表示单位：个十百千...
        for i in range(len(uchars_chinese) - 1, -1, -1):
            val = chinese_num_dict.get(uchars_chinese[i])
            if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                if val > r:
                    r = val
                    total = total + val
                else:
                    r = r * val
            elif val >= 10:
                if val > r:
                    r = val
                else:
                    r = r * val
            else:
                total = total + r * val
        return str(total)

    @staticmethod
    def chinese2digits(chinese_num):
        # 万以上的先忽略？
        # 一个最佳匹配，一个没有单位，一个候选集
        chinese_num_dict = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
        try:
            if chinese_num[-1] in ["万", "亿"]:
                chinese_num = chinese_num[: -1]
            prefix = ""
            if chinese_num[0] == "负":
                chinese_num = chinese_num[1:]
                prefix = "-"
            if "点" in chinese_num:
                index = chinese_num.index("点")
                for i in range(index + 1, len(chinese_num)):
                    a = chinese_num[i]
                    b = chinese_num_dict[a]
                tail = "".join([str(chinese_num_dict[chinese_num[i]]) for i in range(index + 1, len(chinese_num))])
                digit = ValueOptimizer._chinese2digits(chinese_num[: index]) + "." + tail
            else:
                digit = ValueOptimizer._chinese2digits(chinese_num)
            digit = prefix + digit
        except:
            digit = None
        return digit

    @staticmethod
    def create_candidate_set(value):
        candidate_set = set()
        candidate_set.add(value.replace("不限", "是"))
        candidate_set.add(value.replace("达标", "合格"))
        candidate_set.add(value.replace("及格", "合格"))
        candidate_set.add(value.replace("符合", "合格"))
        candidate_set.add(value.replace("达到标准", "合格"))
        candidate_set.add(value.replace("不", "否"))
        candidate_set.add(value.replace("没有", "否"))
        candidate_set.add(value.replace("没有", "未"))
        candidate_set.add(value.replace("不用", "免"))
        candidate_set.add(value.replace("不需要", "免"))
        candidate_set.add(value.replace("没有要求", "不限"))
        candidate_set.add(value.replace("广东话", "粤语"))
        candidate_set.add(value.replace("白话", "粤语"))
        candidate_set.add(value.replace("中大", "中山大学"))
        candidate_set.add(value.replace("重大", "重庆大学"))
        candidate_set.add(value.replace("人大", "中国人民大学"))
        candidate_set.add(value.replace("北大", "北京大学"))
        candidate_set.add(value.replace("南大", "南京大学"))
        candidate_set.add(value.replace("武大", "武汉大学"))
        candidate_set.add(value.replace("复旦", "复旦大学"))
        candidate_set.add(value.replace("清华", "清华大学"))
        candidate_set.add(value.replace("广大", "广州大学"))
        return candidate_set

class QuestionMatcher:
    @staticmethod
    def num2chinese(num):
        num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
                    '0': '零', }
        index_dict = {1: '', 2: '十', 3: '百', 4: '千', 5: '万', 6: '十', 7: '百', 8: '千', 9: '亿'}
        nums = list(num)
        nums_index = [x for x in range(1, len(nums) + 1)][-1::-1]
        chinese_num = ''
        for index, item in enumerate(nums):
            chinese_num = "".join((chinese_num, num_dict[item], index_dict[nums_index[index]]))
        chinese_num = re.sub("零[十百千零]*", "零", chinese_num)
        chinese_num = re.sub("零万", "万", chinese_num)
        chinese_num = re.sub("亿万", "亿零", chinese_num)
        chinese_num = re.sub("零零", "零", chinese_num)
        chinese_num = re.sub("零\\b", "", chinese_num)
        if chinese_num[:2] == "一十":
            chinese_num = chinese_num[1:]
        if num == "0":
            chinese_num = "零"
        return chinese_num

    @staticmethod
    def process_two(chinese_num):
        final_list = []

        def recursive(chinese_num_list, index):
            if index == len(chinese_num_list): return
            if chinese_num_list[index] != "二":
                recursive(chinese_num_list, index + 1)
            else:
                new_chinese_num_list = copy.copy(chinese_num_list)
                new_chinese_num_list[index] = "两"
                final_list.append(chinese_num_list)
                final_list.append(new_chinese_num_list)
                recursive(chinese_num_list, index + 1)
                recursive(new_chinese_num_list, index + 1)

        if "二" in chinese_num:
            recursive(list(chinese_num), 0)
            chinese_nums = list(set(map(lambda x: "".join(x), final_list)))
        else:
            chinese_nums = [chinese_num]
        return chinese_nums

    @staticmethod
    def float2chinese(num):
        num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
                    '0': '零', }
        chinese_num_set = set()
        if num.count(".") == 1 and num[-1] != "." and num[0] != ".":
            index = num.index(".")
            part1 = num[: index]
            part2 = num[index + 1:]
            part1_chinese = QuestionMatcher.num2chinese(part1)
            part2_chinese = "".join(list(map(lambda x: num_dict[x], list(part2))))
            chinese_num_set.add(part1_chinese + "点" + part2_chinese)
            chinese_num_set.add(part1_chinese + "块" + part2_chinese)
            chinese_num_set.add(part1_chinese + "元" + part2_chinese)
            if part1 == "0":
                chinese_num_set.add(part2_chinese)
        else:
            chinese_num_set.add(QuestionMatcher.num2chinese(num.replace(".", "")))
        return chinese_num_set

    @staticmethod
    def create_mix_num(num):
        num = int(num)
        if int(num % 1e12) == 0:  # 万亿
            top_digit = str(int(num / 1e12))
            num = top_digit + "万亿"
        elif int(num % 1e11) == 0:
            top_digit = str(int(num / 1e11))
            num = top_digit + "千亿"
        elif int(num % 1e10) == 0:
            top_digit = str(int(num / 1e10))
            num = top_digit + "百亿"
        elif int(num % 1e9) == 0:
            top_digit = str(int(num / 1e9))
            num = top_digit + "十亿"
        elif int(num % 1e8) == 0:
            top_digit = str(int(num / 1e8))
            num = top_digit + "亿"
        elif int(num % 1e7) == 0:
            top_digit = str(int(num / 1e7))
            num = top_digit + "千万"
        elif int(num % 1e6) == 0:
            top_digit = str(int(num / 1e6))
            num = top_digit + "百万"
        elif int(num % 1e5) == 0:
            top_digit = str(int(num / 1e5))
            num = top_digit + "十万"
        elif int(num % 1e4) == 0:
            top_digit = str(int(num / 1e4))
            num = top_digit + "万"
        elif int(num % 1e3) == 0:
            top_digit = str(int(num / 1e3))
            num = top_digit + "千"
        elif int(num % 1e2) == 0:
            top_digit = str(int(num / 1e2))
            num = top_digit + "百"
        elif int(num % 1e1) == 0:
            top_digit = str(int(num / 1e1))
            num = top_digit + "十"
        else:
            num = str(num)
        return num

    @staticmethod
    def nums_add_unit(nums):
        final_nums = set()
        for num in nums:
            final_nums.add(num)
            final_nums.add(num + "百")
            final_nums.add(num + "千")
            final_nums.add(num + "万")
            final_nums.add(num + "亿")
            if len(num) == 1:
                final_nums.add(num + "十万")
                final_nums.add(num + "百万")
                final_nums.add(num + "千万")
                final_nums.add(num + "十亿")
                final_nums.add(num + "百亿")
                final_nums.add(num + "千亿")
                final_nums.add(num + "万亿")
        final_nums = list(final_nums)
        return final_nums

    @staticmethod
    def num2year(num):
        num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
                    '0': '零', }
        year_list = []
        if "." not in num:
            if len(num) == 4 and 1000 < int(num) < 2100:
                year_num_list = []
                year_num_list.append(num)
                year_num_list.append(num[2] + num[3])
                year_num_list.append(num_dict[num[2]] + num_dict[num[3]])
                year_num_list.append(num_dict[num[0]] + num_dict[num[1]] + num_dict[num[2]] + num_dict[num[3]])
                for year_num in year_num_list:
                    year_list.append(year_num)
                    year_list.append(year_num + "年")
                    year_list.append(year_num + "级")
                    year_list.append(year_num + "届")
            if len(num) == 8 and 1800 < int(num[0: 4]) < 2100 and 0 < int(num[4: 6]) <= 12 and 0 < int(num[6: 8]) <= 31:
                year_list.append("%s年%s月%s日" % (num[0: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
                year_list.append("%s年%s月%s日" % (num[2: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
                year_list.append("%s年%s月%s号" % (num[0: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
                year_list.append("%s年%s月%s号" % (num[2: 4], str(int(num[4: 6])), str(int(num[6: 8]))))
        else:
            if num.count(".") == 1:
                year, month = num.split(".")
                if len(year) >= 2:
                    year_list.append("%s年%s月" % (year, str(int(month))))
                    year_list.append("%s年%s月" % (year[-2:], str(int(month))))
        return year_list

    @staticmethod
    def convert_num(num):
        num = str(num)
        if "." not in num:
            chinese_num = QuestionMatcher.num2chinese(num)
            chinese_nums = QuestionMatcher.process_two(chinese_num)
            mix_num = QuestionMatcher.create_mix_num(num)
            candidate_nums = QuestionMatcher.nums_add_unit(chinese_nums + [num, mix_num])
        else:
            candidate_nums = QuestionMatcher.nums_add_unit([num])
        return candidate_nums

    @staticmethod
    def convert_str(value):
        candidate_substrs = set()
        candidate_substrs.add(value)
        if len(value) > 2:  # 去掉最后一个字
            candidate_substrs.add(value[: -1])
        if ("(" in value and ")" in value) or ("（" in value and "）" in value):
            tmp_value = value.replace("（", "(").replace("）", ")")
            index1 = tmp_value.index("(")
            index2 = tmp_value.index(")")
            if index1 < index2:
                candidate_substrs.add(tmp_value.replace("(", "").replace(")", ""))
                # candidate_substrs.add(tmp_value[index1 + 1: index2])   # 括号不能取出来
                candidate_substrs.add(tmp_value.replace(tmp_value[index1: index2 + 1], ""))
        candidate_substrs.add(value.replace("公司", ""))
        candidate_substrs.add(value.replace("有限", ""))
        candidate_substrs.add(value.replace("有限公司", ""))
        candidate_substrs.add(value.replace("合格", "达标"))
        candidate_substrs.add(value.replace("合格", "及格"))
        candidate_substrs.add(value.replace("不合格", "不达标"))
        candidate_substrs.add(value.replace("不合格", "不及格"))
        candidate_substrs.add(value.replace("风景名胜区", ""))
        candidate_substrs.add(value.replace("著", ""))
        candidate_substrs.add(value.replace("等", ""))
        candidate_substrs.add(value.replace("省", ""))
        candidate_substrs.add(value.replace("市", ""))
        candidate_substrs.add(value.replace("区", ""))
        candidate_substrs.add(value.replace("县", ""))
        candidate_substrs.add(value.replace("岗", "员"))
        candidate_substrs.add(value.replace("员", "岗"))
        candidate_substrs.add(value.replace("岗", "人员"))
        candidate_substrs.add(value.replace("岗", ""))
        candidate_substrs.add(value.replace("人员", "岗"))
        candidate_substrs.add(value.replace("岗位", "人员"))
        candidate_substrs.add(value.replace("人员", "岗位"))
        candidate_substrs.add(value.replace("岗位", ""))
        candidate_substrs.add(value.replace("人员", ""))
        candidate_substrs.add(value.lower())
        candidate_substrs.add(value.replace("-", ""))
        candidate_substrs.add(value.replace("-", "到"))
        candidate_substrs.add(value.replace("-", "至"))
        candidate_substrs.add(value.replace("否", "不"))
        candidate_substrs.add(value.replace("否", "没有"))
        candidate_substrs.add(value.replace("未", "没有"))
        candidate_substrs.add(value.replace("《", "").replace("》", "").replace("<", "").replace(">", ""))
        candidate_substrs.add(value.replace("免费", "免掉"))
        candidate_substrs.add(value.replace("免费", "免"))
        candidate_substrs.add(value.replace("免", "不用"))
        candidate_substrs.add(value.replace("免", "不需要"))
        candidate_substrs.add(value.replace("的", ""))
        candidate_substrs.add(value.replace("\"", "").replace("“", "").replace("”", ""))
        candidate_substrs.add(value.replace("类", ""))
        candidate_substrs.add(value.replace("级", "等"))
        candidate_substrs.add(value.replace("附属小学", "附小"))
        candidate_substrs.add(value.replace("附属中学", "附中"))
        candidate_substrs.add(value.replace("三甲", "三级甲等"))
        candidate_substrs.add(value.replace("三乙", "三级乙等"))
        candidate_substrs.add(value.replace("不限", "不要求"))
        candidate_substrs.add(value.replace("不限", "没有要求"))
        candidate_substrs.add(value.replace("全日制博士", "博士"))
        candidate_substrs.add(value.replace("本科及以上", "本科"))
        candidate_substrs.add(value.replace("硕士及以上学位", "硕士"))
        candidate_substrs.add(value.replace("主编", ""))
        candidate_substrs.add(value.replace("性", ""))
        candidate_substrs.add(value.replace("教师", "老师"))
        candidate_substrs.add(value.replace("老师", "教师"))
        candidate_substrs.add(value.replace(":", ""))
        candidate_substrs.add(value.replace("：", ""))
        candidate_substrs.add(value.replace("股份", "股"))
        candidate_substrs.add(value.replace("股份", ""))
        candidate_substrs.add(value.replace("控股", ""))
        candidate_substrs.add(value.replace("中山大学", "中大"))
        candidate_substrs.add(value.replace("重庆大学", "重大"))
        candidate_substrs.add(value.replace("中国人民大学", "人大"))
        candidate_substrs.add(value.replace("北京大学", "北大"))
        candidate_substrs.add(value.replace("南京大学", "南大"))
        candidate_substrs.add(value.replace("武汉大学", "武大"))
        candidate_substrs.add(value.replace("复旦大学", "复旦"))
        candidate_substrs.add(value.replace("清华大学", "清华"))
        candidate_substrs.add(value.replace("广州大学", "广大"))
        candidate_substrs.add(value.replace("北京体育大学", "北体"))
        candidate_substrs.add(value.replace(".00", ""))
        candidate_substrs.add(value.replace(",", ""))
        candidate_substrs.add(value.replace("，", ""))
        candidate_substrs.add(value.replace("0", "零"))
        candidate_substrs.add(value.replace("第", "").replace("学", ""))
        candidate_substrs.add(value.replace("省", "").replace("市", "").replace("第", "").replace("学", ""))
        candidate_substrs.add(value.replace("年", ""))
        candidate_substrs.add(value.replace("粤语", "广东话"))
        candidate_substrs.add(value.replace("粤语", "白话"))
        candidate_substrs.add(value.replace("市", "").replace("医院", "院"))
        candidate_substrs.add(value.replace("研究生/硕士", "硕士"))
        candidate_substrs.add(value.replace("研究生/硕士", "硕士研究生"))
        candidate_substrs.add(value.replace("中医医院", "中医院"))
        candidate_substrs.add(value.replace("医生", "医师"))
        candidate_substrs.add(value.replace("合格", "符合"))
        candidate_substrs.add(value.replace("合格", "达到标准"))
        candidate_substrs.add(value.replace("工学", "工程学"))
        candidate_substrs.add(value.replace("场", "馆"))
        candidate_substrs.add(value.replace("市保", "市级保护单位"))
        candidate_substrs.add(value.replace("市保", "保护单位"))
        candidate_substrs.add(value.replace("经理女", "女经理"))
        candidate_substrs.add(value.replace("大专及以上", "大专"))
        candidate_substrs.add(value.replace("大专及以上", "专科"))
        candidate_substrs.add(value.replace("北京青年报社", "北青报"))
        candidate_substrs.add(value.replace("不限", "没有限制"))
        candidate_substrs.add(value.replace("高级中学", "高中"))
        candidate_substrs.add(value.replace("中共党员", "党员"))
        digit = "".join(re.findall("[0-9.]", value))
        # 有3年及以上相关工作经验 你能告诉我那些要求[三]年相关工作经验，还有要求本科或者本科以上学历的是什么职位吗
        # 2014WTA广州国际女子网球公开赛 你知道在什么地方举办[2014]年的WTA广州国际女子网球公开赛吗
        if len(digit) > 0 and digit.count(".") <= 1 and QuestionMatcher.is_float(digit) and float(digit) < 1e8 and len(
                digit) / len(value) > 0.4:
            candidate_substrs.add(digit)
            chinese_num_set = QuestionMatcher.float2chinese(digit)
            candidate_substrs |= chinese_num_set
        year1 = re.match("[0-9]{4}年", value)
        if year1:
            year1 = year1.string
            candidate_substrs.add(year1)
            candidate_substrs.add(year1[2:])
        year2 = re.match("[0-9]{4}-[0-9]{2}", value)
        if year2:
            year2 = year2.string
            year = year2[0: 4]
            mongth = year2[5: 7] if year2[5] == "1" else year2[6]
            candidate_substrs.add("%s年%s月" % (year, mongth))
            candidate_substrs.add("%s年%s月" % (year[2:], mongth))
        year3 = re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}", value)
        if year3:
            year3 = year3.string
            year = year3[0: 4]
            mongth = year3[5: 7] if year3[5] == "1" else year3[6]
            day = year3[8: 10] if year3[8] == "1" else year3[9]
            candidate_substrs.add("%s年%s月%s日" % (year, mongth, day))
            candidate_substrs.add("%s年%s月%s日" % (year[2:], mongth, day))
            candidate_substrs.add("%s年%s月%s号" % (year, mongth, day))
            candidate_substrs.add("%s年%s月%s号" % (year[2:], mongth, day))
        return list(candidate_substrs)

    @staticmethod
    def duplicate_relative_index(conds):
        value_dict = {}
        duplicate_indices = []
        for _, _, value in conds:
            if value not in value_dict:
                duplicate_indices.append(0)
                value_dict[value] = 1
            else:
                duplicate_indices.append(value_dict[value])
                value_dict[value] += 1
        return duplicate_indices

    @staticmethod
    def is_float(x):
        try:
            float(x)
            return True
        except:
            return False

    @staticmethod
    def match_str(question, value, precision_limit=0.8, recall_limit=0.65, match_type="recall"):
        value_char_dict = {}
        for char in value:
            if char in value_char_dict:
                value_char_dict[char] += 1
            else:
                value_char_dict[char] = 1
        candidate_substrs = []
        matched_str = ""
        for n in range(2, min(len(question), len(value) + 5)):
            for i in range(0, len(question) - n + 1):
                substr = question[i: i + n]
                char_dict = copy.copy(value_char_dict)
                positive_num = 0
                for char in substr:
                    if char in char_dict and char_dict[char] > 0:
                        positive_num += 1
                        char_dict[char] -= 1
                precision = positive_num / len(substr)
                recall = positive_num / len(value)
                if precision == 0 or recall == 0: continue
                candidate_substrs.append([substr, precision, recall])
        if match_type == "recall":
            fully_matched_substrs = list(filter(lambda x: x[2] == 1, candidate_substrs))
            sorted_list = sorted(fully_matched_substrs, key=lambda x: x[1], reverse=True)
            if len(sorted_list) > 0:
                if sorted_list[0][1] > precision_limit:
                    # print(value, question, sorted_list[0])
                    matched_str = sorted_list[0][0]
        if match_type == "precision":
            precise_substrs = list(filter(lambda x: x[1] == 1, candidate_substrs))
            sorted_list = sorted(precise_substrs, key=lambda x: x[2], reverse=True)
            if len(sorted_list) > 0:
                if sorted_list[0][2] > recall_limit:
                    # print(value, question, sorted_list[0])
                    matched_str = sorted_list[0][0]
        return matched_str

    @staticmethod
    def match_value(question, value, duplicate_index):
        pre_stopchars = set("0123456789一二三四五六七八九十")
        post_stopchars = set("0123456789一二三四五六七八九十百千万亿")
        stopwords = {"一下", "一共", "一起", "一并", "一致", "一周", "一共"}
        original_value = value
        if QuestionMatcher.is_float(value) and float(value) < 1e8:  # 数字转中文只能支持到亿
            if float(value) - math.floor(float(value)) == 0:
                value = str(int(float(value)))
            candidate_nums = QuestionMatcher.convert_num(value) if "-" not in value else [value]  # - 是负数
            year_list = QuestionMatcher.num2year(value)
            candidate_values = candidate_nums + year_list + [original_value]
        else:
            if value in question:
                candidate_values = [value]
            else:
                candidate_values = QuestionMatcher.convert_str(value)
                if sum([candidate_value in question for candidate_value in candidate_values]) == 0:
                    matched_str = QuestionMatcher.match_str(question, value, precision_limit=0.8, recall_limit=0.65,
                                                            match_type="recall")
                    if len(matched_str) > 0:
                        candidate_values.append(matched_str)
        matched_value = ""
        matched_index = None
        for value in candidate_values:
            if value in question and len(value) > len(matched_value):
                indices = [i for i in range(len(question)) if question.startswith(value, i)]
                valid_indices = []
                for index in indices:
                    flag = 0
                    if index - 1 >= 0:
                        previsou_char = question[index - 1]
                        if previsou_char in pre_stopchars: flag = 1
                    if index + len(value) < len(question):
                        post_char = question[index + len(value)]
                        if post_char in post_stopchars: flag = 1
                        if question[index] + post_char in stopwords: flag = 1
                    if flag == 1: continue
                    valid_indices.append(index)
                if len(valid_indices) == 1:
                    matched_value = value
                    matched_index = valid_indices[0]
                elif len(valid_indices) > 1 and duplicate_index < len(valid_indices):
                    matched_value = value
                    matched_index = valid_indices[duplicate_index]
        if matched_value != "":
            question = list(question)
            question_1 = "".join(question[: matched_index])
            question_2 = "".join(question[matched_index: matched_index + len(matched_value)])
            question_3 = "".join(question[matched_index + len(matched_value):])
            question = question_1 + "[" + question_2 + "]" + question_3
            # print(original_value, question)
        else:
            # print(original_value, "不匹配", question)
            pass
        return matched_value, matched_index

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

    def forward(self, input_ids, attention_mask, all_masks, header_masks, question_masks, subheader_masks,
                subheader_cls_list, value_masks, cls_index_list, train_dependencies=None):
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
            tag_masks = train_dependencies[0].view(-1) == 1  # 必须要加 view 和 == 1
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

            tag_logits = torch.argmax(F.log_softmax(tag_output, dim=2), dim=2).detach().cpu().numpy().tolist()
            agg_logits = torch.argmax(F.log_softmax(agg_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            connection_logits = torch.argmax(F.log_softmax(connection_output, dim=1),
                                             dim=1).detach().cpu().numpy().tolist()
            con_num_logits = torch.argmax(F.log_softmax(con_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            type_logits = torch.argmax(F.log_softmax(type_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            sel_num_logits = torch.argmax(F.log_softmax(sel_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            where_num_logits = torch.argmax(F.log_softmax(where_num_output, dim=1),
                                            dim=1).detach().cpu().numpy().tolist()
            op_logits = torch.argmax(F.log_softmax(op_output, dim=1), dim=1).detach().cpu().numpy().tolist()

            return tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits

class Trainer:
    def __init__(self, data_dir, epochs=1, batch_size=64, base_batch_size=32, max_len=120, seed=1234, debug=False):
        self.device = torch.device('cuda')
        self.data_dir = data_dir
        self.debug = debug
        self.seed = seed
        self.seed_everything()
        self.max_len = max_len
        self.epochs = epochs
        self.base_batch_size = base_batch_size
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
            self.bert_model_path = "./data/chinese_wwm_L-12_H-768_A-12"
            self.pytorch_bert_path = "./data/chinese_wwm_L-12_H-768_A-12/pytorch_model.bin"
            self.bert_config = BertConfig("./data/chinese_wwm_L-12_H-768_A-12/bert_config.json")

    def read_query(self, query_path):
        '''
        query_path 是带有用户问题的json 文件路径
        '''
        data = []
        with open(query_path, "r") as data_file:
            for line_index, each_line in enumerate(data_file):
                # debug 只读100行即可
                if self.debug and line_index == 100: break
                data.append(json.loads(each_line))
        print(len(data))
        return data

    def read_table(self, table_path):
        '''
        table_path 是对应于问题的存有完整数据库的json文件
        '''
        table = {}
        with open(table_path, "r") as table_file:
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
        duplicate_indices = QuestionMatcher.duplicate_relative_index(where_conditions)
        condition_dict = {}
        for [where_col, where_op, where_value], duplicate_index in zip(where_conditions, duplicate_indices):
            where_value = where_value.strip()
            matched_value, matched_index = QuestionMatcher.match_value(question, where_value, duplicate_index)
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

        SampleFeaturesLables = InputFeaturesLabels()
        # conc_tokens = []
        # tag_masks = []
        # sel_masks = []
        # con_masks = []
        # type_masks = []
        # attention_masks = []
        # header_masks = []
        # question_masks = []
        # value_masks = []
        # connection_labels = []
        # agg_labels = []
        # tag_labels = []
        # con_num_labels = []
        # type_labels = []
        # cls_index_list = []
        # header_question_list = []
        # header_table_id_list = []
        # subheader_cls_list = []
        # subheader_masks = []
        # sel_num_labels = []
        # where_num_labels = []
        # op_labels = []

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
            print(value_dict)
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
            SampleFeaturesLables.conc_tokens.append(connect_inputIDs)
            SampleFeaturesLables.tag_masks.append(sequence_labeling_inputMask)
            SampleFeaturesLables.sel_masks.append(select_column_mask)
            SampleFeaturesLables.con_masks.append(where_conlumn_inputMask)
            SampleFeaturesLables.type_masks.append(type_mask)
            SampleFeaturesLables.attention_masks.append(attention_mask)
            SampleFeaturesLables.connection_labels.append(where_relation_label)
            SampleFeaturesLables.agg_labels.append(select_agg_label)
            SampleFeaturesLables.tag_labels.append(sequence_labeling_label)
            SampleFeaturesLables.con_num_labels.append(where_conlumn_number_label)
            SampleFeaturesLables.type_labels.append(type_label)
            SampleFeaturesLables.cls_index_list.append(firstColumn_CLS_startPosition)
            SampleFeaturesLables.header_question_list.append(question)
            SampleFeaturesLables.header_table_id_list.append(tableID)
            SampleFeaturesLables.header_masks.append(each_column_inputMask)
            SampleFeaturesLables.question_masks.append(question_inputMask)
            SampleFeaturesLables.subheader_cls_list.append(nextColumn_CLS_startPosition)
            SampleFeaturesLables.subheader_masks.append(nextColumn_inputMask)
            SampleFeaturesLables.sel_num_labels.append(select_number_label)
            SampleFeaturesLables.where_num_labels.append(where_number_label)
            SampleFeaturesLables.op_labels.append(op_label)
            SampleFeaturesLables.value_masks.append(value_inputMask)

        # return tag_masks, sel_masks, con_masks, type_masks, attention_masks, connection_labels, agg_labels, tag_labels, con_num_labels, type_labels, cls_index_list, conc_tokens, header_question_list, header_table_id_list, header_masks, question_masks, subheader_cls_list, subheader_masks, sel_num_labels, where_num_labels, op_labels, value_masks, question_
        return SampleFeaturesLables, question_

    def process_sample_test(self, sample, table_dict, bert_tokenizer):
        question = sample["question"]
        table_id = sample["table_id"]
        table_title = table_dict[table_id]["title"]
        table_header_list = table_dict[table_id]["header"]
        table_row_list = table_dict[table_id]["rows"]
        question = question.strip().replace(" ", "")

        col_dict = {header_name: set() for header_name in table_header_list}
        for row in table_row_list:
            for col, value in enumerate(row):
                header_name = table_header_list[col]
                col_dict[header_name].add(str(value))

        # conc_tokens = []
        # attention_masks = []
        # header_masks = []
        # question_masks = []
        # value_masks = []
        # cls_index_list = []
        # subheader_cls_list = []
        # subheader_masks = []
        SampleFeaturesLabels = InputFeaturesLabels()

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

            SampleFeaturesLabels.conc_tokens.append(conc_ids)
            SampleFeaturesLabels.attention_masks.append(attention_mask)
            SampleFeaturesLabels.cls_index_list.append(header_cls_index)
            SampleFeaturesLabels.header_masks.append(header_mask)
            SampleFeaturesLabels.question_masks.append(question_mask)
            SampleFeaturesLabels.subheader_cls_list.append(subheader_cls_index)
            SampleFeaturesLabels.subheader_masks.append(subheader_mask)
            SampleFeaturesLabels.value_masks.append(value_mask)
        SampleFeaturesLabels.type_masks = [1] * len(SampleFeaturesLabels.conc_tokens)

        #return attention_masks, cls_index_list, conc_tokens, header_masks, question_masks, subheader_cls_list, subheader_masks, value_masks, type_masks, question_tokens
        return SampleFeaturesLabels, question_tokens

    def data_iterator(self, mode = "train"):
        """
        sel 列 agg类型
        where 列 逻辑符 值
        where连接符

        问题开头cls：where连接符（或者新模型，所有header拼一起，预测where连接类型？）
        列的开头cls，多任务学习：1、（不选中，sel，where） 2、agg类型（0~5：agg类型，6：不属于sel） 3、逻辑符类型：（0~3：逻辑符类型，4：不属于where）
        问题部分：序列标注，（每一个字的隐层和列开头cls拼接？再拼接列所有字符的avg？），二分类，如果列是where并且是对应value的，标注为1
        """
        if  mode == "train":
            # train: 41522 val: 4396 test: 4086
            print("Loading data, the mode is \"train\", Loading train_set and valid_set")
            train_data_list = self.read_query(self.train_data_path)
            train_table_dict = self.read_table(self.train_table_path)
            valid_data_list = self.read_query(self.valid_data_path)
            valid_table_dict = self.read_table(self.valid_table_path)
            # test_data_list = self.read_query(self.test_data_path)
            # test_table_dict = self.read_table(self.test_table_path)
            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)
            TrainFeaturesLabels = InputFeaturesLabels()
            ValidFeaturesLabels = InputFeaturesLabels()
            # TestFeaturesLabels = InputFeaturesLabels()

            for sample in train_data_list:
                SampleFeaturesforTrain, train_question= self.process_sample(sample, train_table_dict, bert_tokenizer)
                TrainFeaturesLabels.tag_masks.extend(SampleFeaturesforTrain.tag_masks)
                TrainFeaturesLabels.sel_masks.extend(SampleFeaturesforTrain.sel_masks)
                TrainFeaturesLabels.con_masks.extend(SampleFeaturesforTrain.con_masks)
                TrainFeaturesLabels.type_masks.extend(SampleFeaturesforTrain.type_masks)
                TrainFeaturesLabels.attention_masks.extend(SampleFeaturesforTrain.attention_masks)
                TrainFeaturesLabels.connection_labels.extend(SampleFeaturesforTrain.connection_labels)
                TrainFeaturesLabels.agg_labels.extend(SampleFeaturesforTrain.agg_labels)
                TrainFeaturesLabels.tag_labels.extend(SampleFeaturesforTrain.tag_labels)
                TrainFeaturesLabels.con_num_labels.extend(SampleFeaturesforTrain.con_num_labels)
                TrainFeaturesLabels.type_labels.extend(SampleFeaturesforTrain.type_labels)
                TrainFeaturesLabels.cls_index_list.extend(SampleFeaturesforTrain.cls_index_list)
                TrainFeaturesLabels.conc_tokens.extend(SampleFeaturesforTrain.conc_tokens)
                TrainFeaturesLabels.header_question_list.extend(SampleFeaturesforTrain.header_question_list)
                TrainFeaturesLabels.header_table_id_list.extend(SampleFeaturesforTrain.header_table_id_list)
                TrainFeaturesLabels.header_masks.extend(SampleFeaturesforTrain.header_masks)
                TrainFeaturesLabels.question_masks.extend(SampleFeaturesforTrain.question_masks)
                TrainFeaturesLabels.subheader_cls_list.extend(SampleFeaturesforTrain.subheader_cls_list)
                TrainFeaturesLabels.subheader_masks.extend(SampleFeaturesforTrain.subheader_masks)
                TrainFeaturesLabels.sel_num_labels.extend(SampleFeaturesforTrain.sel_num_labels)
                TrainFeaturesLabels.where_num_labels.extend(SampleFeaturesforTrain.where_num_labels)
                TrainFeaturesLabels.op_labels.extend(SampleFeaturesforTrain.op_labels)
                TrainFeaturesLabels.value_masks.extend(SampleFeaturesforTrain.value_masks)
                TrainFeaturesLabels.question_token_list.append(train_question)
                TrainFeaturesLabels.sample_index_list.append(len(TrainFeaturesLabels.conc_tokens))
                TrainFeaturesLabels.sql_list.append(sample["sql"])
                TrainFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TrainFeaturesLabels.table_id_list.append(sample["table_id"])

            for sample in valid_data_list:
                SampleFeaturesforValid, valid_question = self.process_sample(sample, valid_table_dict, bert_tokenizer)
                ValidFeaturesLabels.tag_masks.extend(SampleFeaturesforValid.tag_masks)
                ValidFeaturesLabels.sel_masks.extend(SampleFeaturesforValid.sel_masks)
                ValidFeaturesLabels.con_masks.extend(SampleFeaturesforValid.con_masks)
                ValidFeaturesLabels.type_masks.extend(SampleFeaturesforValid.type_masks)
                ValidFeaturesLabels.attention_masks.extend(SampleFeaturesforValid.attention_masks)
                ValidFeaturesLabels.connection_labels.extend(SampleFeaturesforValid.connection_labels)
                ValidFeaturesLabels.agg_labels.extend(SampleFeaturesforValid.agg_labels)
                ValidFeaturesLabels.tag_labels.extend(SampleFeaturesforValid.tag_labels)
                ValidFeaturesLabels.con_num_labels.extend(SampleFeaturesforValid.con_num_labels)
                ValidFeaturesLabels.type_labels.extend(SampleFeaturesforValid.type_labels)
                ValidFeaturesLabels.cls_index_list.extend(SampleFeaturesforValid.cls_index_list)
                ValidFeaturesLabels.conc_tokens.extend(SampleFeaturesforValid.conc_tokens)
                ValidFeaturesLabels.header_question_list.extend(SampleFeaturesforValid.header_question_list)
                ValidFeaturesLabels.header_table_id_list.extend(SampleFeaturesforValid.header_table_id_list)
                ValidFeaturesLabels.header_masks.extend(SampleFeaturesforValid.header_masks)
                ValidFeaturesLabels.question_masks.extend(SampleFeaturesforValid.question_masks)
                ValidFeaturesLabels.subheader_cls_list.extend(SampleFeaturesforValid.subheader_cls_list)
                ValidFeaturesLabels.subheader_masks.extend(SampleFeaturesforValid.subheader_masks)
                ValidFeaturesLabels.sel_num_labels.extend(SampleFeaturesforValid.sel_num_labels)
                ValidFeaturesLabels.where_num_labels.extend(SampleFeaturesforValid.where_num_labels)
                ValidFeaturesLabels.op_labels.extend(SampleFeaturesforValid.op_labels)
                ValidFeaturesLabels.value_masks.extend(SampleFeaturesforValid.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question)
                ValidFeaturesLabels.sample_index_list.append(len(ValidFeaturesLabels.conc_tokens))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            train_dataset = data.TensorDataset(torch.tensor(TrainFeaturesLabels.conc_tokens, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.tag_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.sel_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.con_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.type_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.attention_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.connection_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.agg_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.tag_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.con_num_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.type_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.cls_index_list, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.header_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.subheader_cls_list, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.subheader_masks, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.sel_num_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.where_num_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.op_labels, dtype=torch.long),
                                               torch.tensor(TrainFeaturesLabels.value_masks, dtype=torch.long)
                                               )
            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.conc_tokens, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.tag_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.con_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.type_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.connection_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.agg_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.tag_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.con_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.type_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.cls_index_list, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.header_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.subheader_cls_list, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.subheader_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.op_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )
            # 将 dataset 转成 dataloader
            train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=self.base_batch_size, shuffle=True)
            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)
            # test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.base_batch_size, shuffle=False)
            # 返回训练数据
            #return train_iterator, valid_iterator, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list, test_iterator, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict, valid_question_token_list, test_question_token_list
            return train_iterator, valid_iterator, ValidFeaturesLabels, valid_table_dict #, test_iterator, TestFeaturesLabels, test_table_dict
        elif mode == "test":
            print("Loading data, the mode is \"test\", Loading test_set")
            test_data_list = self.read_query(self.test_data_path)
            test_table_dict = self.read_table(self.test_table_path)

            TestFeaturesLabels = InputFeaturesLabels()

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            for sample in test_data_list:
                FeaturesLabelsforTest, test_question = self.process_sample_test(sample, test_table_dict, bert_tokenizer)
                TestFeaturesLabels.attention_masks.extend(FeaturesLabelsforTest.attention_masks)
                TestFeaturesLabels.cls_index_list.extend(FeaturesLabelsforTest.cls_index_list)
                TestFeaturesLabels.conc_tokens.extend(FeaturesLabelsforTest.conc_tokens)
                TestFeaturesLabels.header_masks.extend(FeaturesLabelsforTest.header_masks)
                TestFeaturesLabels.question_masks.extend(FeaturesLabelsforTest.question_masks)
                TestFeaturesLabels.subheader_cls_list.extend(FeaturesLabelsforTest.subheader_cls_list)
                TestFeaturesLabels.subheader_masks.extend(FeaturesLabelsforTest.subheader_masks)
                TestFeaturesLabels.value_masks.extend(FeaturesLabelsforTest.value_masks)
                TestFeaturesLabels.type_masks.extend(FeaturesLabelsforTest.type_masks)
                TestFeaturesLabels.question_token_list.append(test_question)
                TestFeaturesLabels.sample_index_list.append(len(TestFeaturesLabels.conc_tokens))
                TestFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TestFeaturesLabels.table_id_list.append(sample["table_id"])

            test_dataset = data.TensorDataset(torch.tensor(TestFeaturesLabels.conc_tokens, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.attention_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.cls_index_list, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.header_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.question_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.subheader_cls_list, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.subheader_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.value_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.type_masks, dtype=torch.long),
                                              )

            test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.base_batch_size, shuffle=False)

            return test_iterator, TestFeaturesLabels, test_table_dict
        elif mode == "evaluate":
            print("Loading data, the mode is \"evaluate\", Loading valid_set")
            valid_data_list = self.read_query(self.valid_data_path)
            valid_table_dict = self.read_table(self.valid_table_path)

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            ValidFeaturesLabels = InputFeaturesLabels()

            for sample in valid_data_list:
                SampleFeaturesforValid, valid_question = self.process_sample(sample, valid_table_dict, bert_tokenizer)
                ValidFeaturesLabels.tag_masks.extend(SampleFeaturesforValid.tag_masks)
                ValidFeaturesLabels.sel_masks.extend(SampleFeaturesforValid.sel_masks)
                ValidFeaturesLabels.con_masks.extend(SampleFeaturesforValid.con_masks)
                ValidFeaturesLabels.type_masks.extend(SampleFeaturesforValid.type_masks)
                ValidFeaturesLabels.attention_masks.extend(SampleFeaturesforValid.attention_masks)
                ValidFeaturesLabels.connection_labels.extend(SampleFeaturesforValid.connection_labels)
                ValidFeaturesLabels.agg_labels.extend(SampleFeaturesforValid.agg_labels)
                ValidFeaturesLabels.tag_labels.extend(SampleFeaturesforValid.tag_labels)
                ValidFeaturesLabels.con_num_labels.extend(SampleFeaturesforValid.con_num_labels)
                ValidFeaturesLabels.type_labels.extend(SampleFeaturesforValid.type_labels)
                ValidFeaturesLabels.cls_index_list.extend(SampleFeaturesforValid.cls_index_list)
                ValidFeaturesLabels.conc_tokens.extend(SampleFeaturesforValid.conc_tokens)
                ValidFeaturesLabels.header_question_list.extend(SampleFeaturesforValid.header_question_list)
                ValidFeaturesLabels.header_table_id_list.extend(SampleFeaturesforValid.header_table_id_list)
                ValidFeaturesLabels.header_masks.extend(SampleFeaturesforValid.header_masks)
                ValidFeaturesLabels.question_masks.extend(SampleFeaturesforValid.question_masks)
                ValidFeaturesLabels.subheader_cls_list.extend(SampleFeaturesforValid.subheader_cls_list)
                ValidFeaturesLabels.subheader_masks.extend(SampleFeaturesforValid.subheader_masks)
                ValidFeaturesLabels.sel_num_labels.extend(SampleFeaturesforValid.sel_num_labels)
                ValidFeaturesLabels.where_num_labels.extend(SampleFeaturesforValid.where_num_labels)
                ValidFeaturesLabels.op_labels.extend(SampleFeaturesforValid.op_labels)
                ValidFeaturesLabels.value_masks.extend(SampleFeaturesforValid.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question)
                ValidFeaturesLabels.sample_index_list.append(len(ValidFeaturesLabels.conc_tokens))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.conc_tokens, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.tag_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.con_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.type_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.connection_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.agg_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.tag_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.con_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.type_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.cls_index_list, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.header_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.subheader_cls_list, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.subheader_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.op_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )

            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)

            return valid_iterator, ValidFeaturesLabels, valid_table_dict #, test_iterator, TestFeaturesLabels, test_table_dict
        elif mode == "test&evaluate":
            print("Loading data, the mode is \"test&evaluate\", Loading test_set and valid_set")
            test_data_list = self.read_query(self.test_data_path)
            test_table_dict = self.read_table(self.test_table_path)
            valid_data_list = self.read_query(self.valid_data_path)
            valid_table_dict = self.read_table(self.valid_table_path)

            TestFeaturesLabels = InputFeaturesLabels()
            ValidFeaturesLabels = InputFeaturesLabels()

            bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_path, cache_dir=None, do_lower_case=True)

            for sample in valid_data_list:
                SampleFeaturesforValid, valid_question = self.process_sample(sample, valid_table_dict, bert_tokenizer)
                ValidFeaturesLabels.tag_masks.extend(SampleFeaturesforValid.tag_masks)
                ValidFeaturesLabels.sel_masks.extend(SampleFeaturesforValid.sel_masks)
                ValidFeaturesLabels.con_masks.extend(SampleFeaturesforValid.con_masks)
                ValidFeaturesLabels.type_masks.extend(SampleFeaturesforValid.type_masks)
                ValidFeaturesLabels.attention_masks.extend(SampleFeaturesforValid.attention_masks)
                ValidFeaturesLabels.connection_labels.extend(SampleFeaturesforValid.connection_labels)
                ValidFeaturesLabels.agg_labels.extend(SampleFeaturesforValid.agg_labels)
                ValidFeaturesLabels.tag_labels.extend(SampleFeaturesforValid.tag_labels)
                ValidFeaturesLabels.con_num_labels.extend(SampleFeaturesforValid.con_num_labels)
                ValidFeaturesLabels.type_labels.extend(SampleFeaturesforValid.type_labels)
                ValidFeaturesLabels.cls_index_list.extend(SampleFeaturesforValid.cls_index_list)
                ValidFeaturesLabels.conc_tokens.extend(SampleFeaturesforValid.conc_tokens)
                ValidFeaturesLabels.header_question_list.extend(SampleFeaturesforValid.header_question_list)
                ValidFeaturesLabels.header_table_id_list.extend(SampleFeaturesforValid.header_table_id_list)
                ValidFeaturesLabels.header_masks.extend(SampleFeaturesforValid.header_masks)
                ValidFeaturesLabels.question_masks.extend(SampleFeaturesforValid.question_masks)
                ValidFeaturesLabels.subheader_cls_list.extend(SampleFeaturesforValid.subheader_cls_list)
                ValidFeaturesLabels.subheader_masks.extend(SampleFeaturesforValid.subheader_masks)
                ValidFeaturesLabels.sel_num_labels.extend(SampleFeaturesforValid.sel_num_labels)
                ValidFeaturesLabels.where_num_labels.extend(SampleFeaturesforValid.where_num_labels)
                ValidFeaturesLabels.op_labels.extend(SampleFeaturesforValid.op_labels)
                ValidFeaturesLabels.value_masks.extend(SampleFeaturesforValid.value_masks)
                ValidFeaturesLabels.question_token_list.append(valid_question)
                ValidFeaturesLabels.sample_index_list.append(len(ValidFeaturesLabels.conc_tokens))
                ValidFeaturesLabels.sql_list.append(sample["sql"])
                ValidFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                ValidFeaturesLabels.table_id_list.append(sample["table_id"])

            for sample in test_data_list:
                FeaturesLabelsforTest, test_question = self.process_sample_test(sample, test_table_dict, bert_tokenizer)
                TestFeaturesLabels.attention_masks.extend(FeaturesLabelsforTest.attention_masks)
                TestFeaturesLabels.cls_index_list.extend(FeaturesLabelsforTest.cls_index_list)
                TestFeaturesLabels.conc_tokens.extend(FeaturesLabelsforTest.conc_tokens)
                TestFeaturesLabels.header_masks.extend(FeaturesLabelsforTest.header_masks)
                TestFeaturesLabels.question_masks.extend(FeaturesLabelsforTest.question_masks)
                TestFeaturesLabels.subheader_cls_list.extend(FeaturesLabelsforTest.subheader_cls_list)
                TestFeaturesLabels.subheader_masks.extend(FeaturesLabelsforTest.subheader_masks)
                TestFeaturesLabels.value_masks.extend(FeaturesLabelsforTest.value_masks)
                TestFeaturesLabels.type_masks.extend(FeaturesLabelsforTest.type_masks)
                TestFeaturesLabels.question_token_list.append(test_question)
                TestFeaturesLabels.sample_index_list.append(len(TestFeaturesLabels.conc_tokens))
                TestFeaturesLabels.question_list.append(sample["question"].strip().replace(" ", ""))
                TestFeaturesLabels.table_id_list.append(sample["table_id"])

            valid_dataset = data.TensorDataset(torch.tensor(ValidFeaturesLabels.conc_tokens, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.tag_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.con_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.type_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.attention_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.connection_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.agg_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.tag_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.con_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.type_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.cls_index_list, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.header_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.question_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.subheader_cls_list, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.subheader_masks, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.sel_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.where_num_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.op_labels, dtype=torch.long),
                                               torch.tensor(ValidFeaturesLabels.value_masks, dtype=torch.long)
                                               )
            test_dataset = data.TensorDataset(torch.tensor(TestFeaturesLabels.conc_tokens, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.attention_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.cls_index_list, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.header_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.question_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.subheader_cls_list, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.subheader_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.value_masks, dtype=torch.long),
                                              torch.tensor(TestFeaturesLabels.type_masks, dtype=torch.long),
                                              )

            valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=self.base_batch_size, shuffle=False)
            test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=self.base_batch_size, shuffle=False)


            return valid_iterator, ValidFeaturesLabels, valid_table_dict, test_iterator, TestFeaturesLabels, test_table_dict
        else:
            print("There is no such mode for data_iterator, please select mode from \"train, test, evaluate, test&evaluate\"")



    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))
    #
    # def detail_score(self, y_true, y_pred, num_labels, ignore_num=None):
    #     detail_y_true = [[] for _ in range(num_labels)]
    #     detail_y_pred = [[] for _ in range(num_labels)]
    #     for i in range(len(y_pred)):
    #         for label in range(num_labels):
    #             if y_true[i] == label:
    #                 detail_y_true[label].append(1)
    #             else:
    #                 detail_y_true[label].append(0)
    #             if y_pred[i] == label:
    #                 detail_y_pred[label].append(1)
    #             else:
    #                 detail_y_pred[label].append(0)
    #     pre_list = []
    #     rec_list = []
    #     f1_list = []
    #     detail_output_str = ""
    #     for label in range(num_labels):
    #         if label == ignore_num: continue
    #         pre = precision_score(detail_y_true[label], detail_y_pred[label])
    #         rec = recall_score(detail_y_true[label], detail_y_pred[label])
    #         f1 = f1_score(detail_y_true[label], detail_y_pred[label])
    #         detail_output_str += "[%d] pre:%.3f rec:%.3f f1:%.3f\n" % (label, pre, rec, f1)
    #         pre_list.append(pre)
    #         rec_list.append(rec)
    #         f1_list.append(f1)
    #     acc = accuracy_score(y_true, y_pred)
    #     output_str = "overall_acc:%.3f, avg_pre:%.3f, avg_rec:%.3f, avg_f1:%.3f \n" % (
    #     acc, np.mean(pre_list), np.mean(rec_list), np.mean(f1_list))
    #     output_str += detail_output_str
    #     return output_str
    #
    # def sql_match(self, s1, s2):
    #     return (s1['cond_conn_op'] == s2['cond_conn_op']) & \
    #            (set(zip(s1['sel'], s1['agg'])) == set(zip(s2['sel'], s2['agg']))) & \
    #            (set([tuple(i) for i in s1['conds']]) == set([tuple(i) for i in s2['conds']]))
    #
    # def evaluate(self, logits_lists, cls_index_list, labels_lists, question_list, question_token_list, table_id_list,
    #              sample_index_list, correct_sql_list, table_dict, header_question_list, header_table_id_list,
    #              do_test=False):
    #     [
    #         sequence_labeling_predict,
    #         select_agg_predict,
    #         where_relation_predict,
    #         where_conlumn_number_predict,
    #         type_logits_list,
    #         sel_num_logits_list,
    #         where_num_logits_list,
    #         type_probs_list,
    #         op_logits_list
    #     ] = logits_lists
    #     [
    #         tag_labels_list,
    #         agg_labels_list,
    #         connection_labels_list,
    #         con_num_labels_list,
    #         type_labels_list,
    #         sel_num_labels_list,
    #         where_num_labels_list,
    #         op_labels_list
    #     ] = labels_lists
    #
    #     # print(tag_logits_list)
    #     # agg_logits_list,
    #     # connection_logits_list,
    #     # con_num_logits_list,
    #     # type_logits_list,
    #     # sel_num_logits_list,
    #     # where_num_logits_list,
    #     # type_probs_list,
    #     # op_logits_list
    #     # exit()
    #     f_valid = open("valid_detail.txt", 'w')
    #     # {"agg": [0], "cond_conn_op": 2, "sel": [1], "conds": [[3, 0, "11"], [6, 0, "11"]]}
    #     sql_dict = {"agg": [], "cond_conn_op": None, "sel": [], "conds": []}
    #     sql_list = []
    #     matched_num = 0
    #     for i in range(len(sample_index_list)):
    #         start_index = 0 if i == 0 else sample_index_list[i - 1]
    #         end_index = sample_index_list[i]
    #         sample_question = question_list[i]
    #         sample_question_token = question_token_list[i]
    #         sample_table_id = table_id_list[i]
    #         if do_test is False:
    #             sample_sql = correct_sql_list[i]
    #         sample_tag_logits = sequence_labeling_predict[start_index: end_index]
    #         sample_agg_logits = select_agg_predict[start_index: end_index]
    #         sample_connection_logits = where_relation_predict[start_index: end_index]
    #         sample_con_num_logits = where_conlumn_number_predict[start_index: end_index]
    #         sample_type_logits = type_logits_list[start_index: end_index]
    #         sample_sel_num_logits = sel_num_logits_list[start_index: end_index]
    #         sample_where_num_logits = where_num_logits_list[start_index: end_index]
    #         sample_op_logits_list = op_logits_list[start_index: end_index]
    #
    #         cls_index = cls_index_list[start_index]
    #         table_header_list = table_dict[sample_table_id]["header"]
    #         table_type_list = table_dict[sample_table_id]["types"]
    #         table_row_list = table_dict[sample_table_id]["rows"]
    #         col_dict = {i: [] for i in range(len(table_header_list))}
    #         for row in table_row_list:
    #             for col, value in enumerate(row):
    #                 col_dict[col].append(str(value))
    #
    #         value_change_list = []
    #         sel_prob_list = []
    #         where_prob_list = []
    #         for j, col_type in enumerate(sample_type_logits):
    #             type_probs = type_probs_list[j]
    #             sel_prob = type_probs[0]
    #             where_prob = type_probs[1]
    #
    #             # sel
    #             agg = sample_agg_logits[j]
    #             sel_col = j
    #             sel_prob_list.append({"prob": sel_prob, "type": col_type, "sel": sel_col, "agg": agg})
    #
    #             # where
    #             tag_list = sample_tag_logits[j][1: cls_index - 1]
    #             con_num = sample_con_num_logits[j]
    #             col_op = sample_op_logits_list[j]
    #             con_col = j
    #             candidate_list = [[[], []]]
    #             candidate_list_index = 0
    #             value_start_index_list = []
    #             previous_tag = -1
    #
    #             # 把 token 的 tag_list 扩展成 question 长度
    #             question_tag_list = []
    #             for i in range(len(tag_list)):
    #                 tag = tag_list[i]
    #                 token = sample_question_token[i]
    #                 token = token.replace("##", "")
    #                 if token == "[UNK]":
    #                     question_tag_list.extend([tag])
    #                 else:
    #                     question_tag_list.extend([tag] * len(token))
    #
    #             for i in range(0, len(question_tag_list)):
    #                 current_tag = question_tag_list[i]
    #                 # 一个 value 结束
    #                 if current_tag == 0:
    #                     if previous_tag == 1:
    #                         candidate_list.append([[], []])
    #                         candidate_list_index += 1
    #                 # 一个 value 开始
    #                 else:
    #                     if previous_tag in [-1, 0]:
    #                         value_start_index_list.append(i)
    #                     candidate_list[candidate_list_index][0].append(sample_question[i])  # 多了一个 cls
    #                     candidate_list[candidate_list_index][1].append(question_tag_list[i])
    #                 previous_tag = current_tag
    #             con_list = []
    #             # for candidate in candidate_list:
    #             for i in range(len(value_start_index_list)):
    #                 candidate = candidate_list[i]
    #                 value_start_index = value_start_index_list[i]
    #                 str_list = candidate[0]
    #                 if len(str_list) == 0: continue
    #                 value_str = "".join(str_list)
    #                 header = table_header_list[j]
    #                 col_data_type = table_type_list[j]
    #                 col_values = col_dict[j]
    #                 op = col_op
    #                 candidate_value_set = set()
    #                 new_value, longest_digit_num, longest_chinese_num = ValueOptimizer.find_longest_num(value_str,
    #                                                                                                     sample_question,
    #                                                                                                     value_start_index)
    #                 candidate_value_set.add(value_str)
    #                 candidate_value_set.add(new_value)
    #                 if longest_digit_num:
    #                     candidate_value_set.add(longest_digit_num)
    #                 digit = None
    #                 if longest_chinese_num:
    #                     candidate_value_set.add(longest_chinese_num)
    #                     digit = ValueOptimizer.chinese2digits(longest_chinese_num)
    #                     if digit:
    #                         candidate_value_set.add(digit)
    #                 replace_candidate_set = ValueOptimizer.create_candidate_set(value_str)
    #                 candidate_value_set |= replace_candidate_set
    #                 # 确定 value 值
    #                 final_value = value_str  # default
    #                 if op != 2:  # 不是 =，不能搜索，能比大小的应该就是数字
    #                     if longest_digit_num:
    #                         final_value = longest_digit_num
    #                         if final_value != value_str: value_change_list.append([value_str, final_value])
    #                     elif digit:
    #                         final_value = digit
    #                         if final_value != value_str: value_change_list.append([value_str, final_value])
    #                 else:
    #                     if value_str not in col_values:
    #                         best_value = ValueOptimizer.select_best_matched_value_from_candidates(
    #                             candidate_value_set, col_values)
    #                         if len(best_value) > 0:
    #                             final_value = best_value
    #                             if final_value != value_str: value_change_list.append([value_str, final_value])
    #                         else:
    #                             value_change_list.append([value_str, "丢弃"])
    #                             continue  # =，不在列表内，也没找到模糊匹配，抛弃
    #                 # con_list 是一列里面的 con
    #                 con_list.append([con_col, op, final_value])
    #             if len(con_list) == con_num:
    #                 for [con_col, op, final_value] in con_list:
    #                     where_prob_list.append(
    #                         {"prob": where_prob, "type": col_type, "cond": [con_col, op, final_value]})
    #             else:
    #                 if len(con_list) > 0:
    #                     [con_col, op, final_value] = con_list[0]
    #                     where_prob_list.append(
    #                         {"prob": where_prob, "type": col_type, "cond": [con_col, op, final_value]})
    #         sel_num = max(sample_sel_num_logits, key=sample_sel_num_logits.count)
    #         where_num = max(sample_where_num_logits, key=sample_where_num_logits.count)
    #
    #         # connection = max(real_connection_list, key=real_connection_list.count) if where_num > 1 and len(real_connection_list) > 0 else 0
    #         # type_dict = {0: "sel", 1: "con", 2: "none"}
    #         sel_prob_list = sorted(sel_prob_list, key=lambda x: (-x["type"], x["prob"]), reverse=True)
    #         where_prob_list = sorted(where_prob_list, key=lambda x: (-(x["type"] ** 2 - 1) ** 2, x["prob"]),
    #                                  reverse=True)
    #
    #         # TODO: connection只有where时才预测，要改过来，前where
    #         if where_num <= 1 or len(where_prob_list) == 0:
    #             connection = 0
    #         else:
    #             where_cols = list(map(lambda x: x["cond"][0], where_prob_list[: where_num]))
    #             real_connection_list = [sample_connection_logits[k] for k in where_cols]
    #             connection = max(real_connection_list, key=real_connection_list.count)
    #
    #         tmp_sql_dict = copy.deepcopy(sql_dict)
    #         tmp_sql_dict["cond_conn_op"] = connection
    #         for j in range(min(sel_num, len(sel_prob_list))):
    #             tmp_sql_dict["agg"].append(sel_prob_list[j]["agg"])
    #             tmp_sql_dict["sel"].append(sel_prob_list[j]["sel"])
    #         for j in range(min(where_num, len(where_prob_list))):
    #             tmp_sql_dict["conds"].append(where_prob_list[j]["cond"])
    #         sql_list.append(tmp_sql_dict)
    #
    #         if do_test is False:
    #             if self.sql_match(tmp_sql_dict, sample_sql):
    #                 matched_num += 1
    #             else:
    #                 f_valid.write("%s\n" % str(sample_question))
    #                 f_valid.write("%s\n" % str(tmp_sql_dict))
    #                 f_valid.write("%s\n" % str(sample_sql))
    #                 # f_valid.write("%s\n" % str(value_change_list))
    #                 cols = set(map(lambda x: x[0], tmp_sql_dict["conds"])) | set(
    #                     map(lambda x: x[0], sample_sql["conds"]))
    #                 for j, table_header in enumerate(table_header_list):
    #                     if j in cols:
    #                         f_valid.write("%d、%s\n" % (j, table_header))
    #                 f_valid.write("\n")
    #
    #     if do_test is False:
    #         logical_acc = matched_num / len(sample_index_list)
    #         print("logical_acc", logical_acc)
    #
    #         op_sql_dict = {0: ">", 1: "<", 2: "==", 3: "!=", 4: "不选中"}
    #         agg_sql_dict = {0: "", 1: "AVG", 2: "MAX", 3: "MIN", 4: "COUNT", 5: "SUM"}
    #         conn_sql_dict = {0: "", 1: "and", 2: "or"}
    #         con_num_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    #         type_dict = {0: "sel", 1: "con", 2: "none"}
    #
    #         tag_pred = []
    #         tag_true = []
    #         tag_fully_matched = []
    #         agg_pred = []
    #         agg_true = []
    #         connection_pred = []
    #         connection_true = []
    #         con_num_pred = []
    #         con_num_true = []
    #         type_pred = type_logits_list
    #         type_true = type_labels_list
    #         sel_num_pred = sel_num_logits_list
    #         sel_num_true = sel_num_labels_list
    #         where_num_pred = where_num_logits_list
    #         where_num_true = where_num_labels_list
    #         op_pred = []
    #         op_true = []
    #
    #         for i, col_type in enumerate(type_true):
    #             if col_type == 0:  # sel
    #                 agg_pred.append(select_agg_predict[i])
    #                 agg_true.append(agg_labels_list[i])
    #             elif col_type == 1:  # con
    #                 cls_index = cls_index_list[i]
    #                 tmp_tag_pred = sequence_labeling_predict[i][1: cls_index - 1]  # 不取 cls 和 sep
    #                 tmp_tag_true = tag_labels_list[i][1: cls_index - 1]
    #                 question = header_question_list[i]
    #                 table_id = header_table_id_list[i]
    #                 matched = 1 if tmp_tag_pred == tmp_tag_true else 0
    #                 tag_fully_matched.append(matched)
    #                 tag_pred.extend(tmp_tag_pred)
    #                 tag_true.extend(tmp_tag_true)
    #                 connection_pred.append(where_relation_predict[i])
    #                 connection_true.append(connection_labels_list[i])
    #                 con_num_pred.append(where_conlumn_number_predict[i])
    #                 con_num_true.append(con_num_labels_list[i])
    #                 op_pred.append(op_logits_list[i])
    #                 op_true.append(op_labels_list[i])
    #
    #         eval_result = ""
    #         eval_result += "TYPE\n" + self.detail_score(type_true, type_pred, num_labels=3, ignore_num=None) + "\n"
    #         eval_result += "TAG\n" + self.detail_score(tag_true, tag_pred, num_labels=2, ignore_num=None) + "\n"
    #         eval_result += "CONNECTION\n" + self.detail_score(connection_true, connection_pred, num_labels=3,
    #                                                           ignore_num=None) + "\n"
    #         eval_result += "CON_NUM\n" + self.detail_score(con_num_true, con_num_pred, num_labels=4,
    #                                                        ignore_num=0) + "\n"
    #         eval_result += "AGG\n" + self.detail_score(agg_true, agg_pred, num_labels=6, ignore_num=None) + "\n"
    #         eval_result += "SEL_NUM\n" + self.detail_score(sel_num_true, sel_num_pred, num_labels=4,
    #                                                        ignore_num=0) + "\n"
    #         eval_result += "WHERE_NUM\n" + self.detail_score(where_num_true, where_num_pred, num_labels=5,
    #                                                          ignore_num=0) + "\n"
    #         eval_result += "OP\n" + self.detail_score(op_true, op_pred, num_labels=4, ignore_num=None) + "\n"
    #
    #         tag_acc = accuracy_score(tag_true, tag_pred)
    #
    #         return eval_result, tag_acc, logical_acc
    #
    #     else:
    #         f_result = open("result.json", 'w')
    #         for sql_dict in sql_list:
    #             sql_dict_json = json.dumps(sql_dict, ensure_ascii=False)
    #             f_result.write(sql_dict_json + '\n')
    #         f_result.close()

    def train(self, batch_size, base_batch_size):
        self.batch_size = batch_size
        self.base_batch_size = base_batch_size
        print("batch_size is {}, base_batch_size is {}".format(self.batch_size, self.base_batch_size))
        if self.debug: self.epochs = 1
        # 加载 dataloader
        # train_iterator, valid_iterator, valid_question_list, valid_table_id_list, valid_sample_index_list, valid_sql_list, valid_table_dict, valid_header_question_list, valid_header_table_id_list, test_iterator, test_question_list, test_table_id_list, test_sample_index_list, test_table_dict, valid_question_token_list, test_question_token_list = self.data_iterator()
        train_iterator, valid_iterator, ValidFeaturesLabels, valid_table_dict = self.data_iterator(mode="train")
        # 训练
        self.seed_everything()
        lr = 1e-5
        accumulation_steps = math.ceil(self.batch_size / self.base_batch_size)
        # 预训练 bert 转成 pytorch
        # 加载预训练模型
        model = BertNeuralNet.from_pretrained(self.bert_model_path, cache_dir=None)
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
        epoch_steps = int(train_iterator.sampler.num_samples / self.base_batch_size / accumulation_steps)
        num_train_optimization_steps = int(self.epochs * epoch_steps)
        optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.05, t_total=num_train_optimization_steps)
        # 开始训练
        f_log = open("train_log.txt", "w")
        best_score = 0
        model.train()
        for epoch in range(self.epochs):
            train_start_time = time.time()
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
                    cls_index_list = batch_data[11].to(self.device)
                    header_masks = batch_data[12].to(self.device)
                    question_masks = batch_data[13].to(self.device)
                    subheader_cls_list = batch_data[14].to(self.device)
                    subheader_masks = batch_data[15].to(self.device)
                    sel_num_labels = batch_data[16].to(self.device)
                    where_num_labels = batch_data[17].to(self.device)
                    op_labels = batch_data[18].to(self.device)
                    value_masks = batch_data[19].to(self.device)
                if torch.sum(sel_masks) == 0 or torch.sum(con_masks) == 0 or torch.sum(tag_masks) == 0: continue
                train_dependencies = [tag_masks, sel_masks, con_masks, connection_labels, agg_labels, tag_labels,
                                      con_num_labels, type_labels, sel_num_labels, where_num_labels, op_labels]
                loss = model(input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                             subheader_cls_list, value_masks, cls_index_list, train_dependencies=train_dependencies)
                loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            # 开始验证
            valid_start_time = time.time()
            model.eval()
            sequence_labeling_predict = []
            select_agg_predict = []
            where_relation_predict = []
            where_conlumn_number_predict = []
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

                sequence_labeling_predict.extend(tag_logits)
                select_agg_predict.extend(agg_logits)
                where_relation_predict.extend(connection_logits)
                where_conlumn_number_predict.extend(con_num_logits)
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

            logits_lists = [sequence_labeling_predict, select_agg_predict, where_relation_predict,
                            where_conlumn_number_predict, type_logits_list, sel_num_logits_list, where_num_logits_list,
                            type_probs_list, op_logits_list]
            labels_lists = [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list,
                            type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list]
            # eval_result, tag_acc, logical_acc = self.evaluate(logits_lists, cls_index_list, labels_lists,
            #                                                   ValidFeaturesLabels.question_list, ValidFeaturesLabels.token_list,
            #                                                   ValidFeaturesLabels.table_id_list, ValidFeaturesLabels.sample_index_list,
            #                                                   ValidFeaturesLabels.sql_list, valid_table_dict,
            #                                                   ValidFeaturesLabels.header_question_list, ValidFeaturesLabels.header_table_id_list)

            eval_result, tag_acc, logical_acc = Evaluate.evaluate(ValueOptimizer, logits_lists, cls_index_list, labels_lists,
                                                            ValidFeaturesLabels.question_list, ValidFeaturesLabels.token_list,
                                                            ValidFeaturesLabels.table_id_list, ValidFeaturesLabels.sample_index_list,
                                                            ValidFeaturesLabels.sql_list, valid_table_dict,
                                                            ValidFeaturesLabels.header_question_list, ValidFeaturesLabels.header_table_id_list)

            score = logical_acc
            # print("epoch: %d duration: %d min \n" % (epoch + 1, int((time.time() - train_start_time) / 60)))
            print("epoch: %d, train_duration: %d min , valid_duration: %d min \n" % (
            epoch + 1, int((valid_start_time - train_start_time) / 60), int((time.time() - valid_start_time) / 60)))
            print(eval_result)
            f_log.write("epoch: %d, train_duration: %d min , valid_duration: %d min \n" % (
            epoch + 1, int((valid_start_time - train_start_time) / 60), int((time.time() - valid_start_time) / 60)))
            f_log.write("\nOVERALL\nlogical_acc: %.3f, tag_acc: %.3f\n\n" % (logical_acc, tag_acc))
            f_log.write(eval_result + "\n")
            f_log.flush()
            save_start_time = time.time()

            if not self.debug and score > best_score:
                best_score = score
                state_dict = model.state_dict()
                model_name = "my_model.bin"
                torch.save(state_dict, model_name)
                print("model save duration: %d min" % int((time.time() - save_start_time) / 60))
                f_log.write("model save duration: %d min\n" % int((time.time() - save_start_time) / 60))

            model.train()
        f_log.close()
        # del 训练相关输入和模型
        training_history = [train_iterator, valid_iterator, model, optimizer, param_optimizer,
                            optimizer_grouped_parameters]
        for variable in training_history:
            del variable
        gc.collect()

    def test(self, batch_size, base_batch_size, do_evaluate=True, do_test=True):
        self.batch_size = batch_size
        self.base_batch_size = base_batch_size
        print("batch_size is {}, base_batch_size is {}".format(self.batch_size, self.base_batch_size))
        # print('load data')
        # train_iterator, valid_iterator, ValidFeaturesLabels, valid_table_dict, test_iterator, TestFeaturesLabels, test_table_dict = self.data_iterator()
        self.seed_everything()
        model = BertNeuralNet(self.bert_config)
        model.load_state_dict(torch.load("./my_model.bin"))
        model = model.to(self.device) if torch.cuda.is_available() else model
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        if do_evaluate:
            valid_iterator, ValidFeaturesLabels, valid_table_dict = self.data_iterator(mode="evaluate")
            sequence_labeling_predict = []
            select_agg_predict = []
            where_relation_predict = []
            where_conlumn_number_predict = []
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

                sequence_labeling_predict.extend(tag_logits)
                select_agg_predict.extend(agg_logits)
                where_relation_predict.extend(connection_logits)
                where_conlumn_number_predict.extend(con_num_logits)
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

            logits_lists = [sequence_labeling_predict, select_agg_predict, where_relation_predict,
                            where_conlumn_number_predict, type_logits_list, sel_num_logits_list, where_num_logits_list,
                            type_probs_list, op_logits_list]
            labels_lists = [tag_labels_list, agg_labels_list, connection_labels_list, con_num_labels_list,
                            type_labels_list, sel_num_labels_list, where_num_labels_list, op_labels_list]
            eval_result, tag_acc, logical_acc = Evaluate.evaluate(ValueOptimizer, logits_lists, cls_index_list, labels_lists,
                                                            ValidFeaturesLabels.question_list, ValidFeaturesLabels.token_list,
                                                            ValidFeaturesLabels.table_id_list, ValidFeaturesLabels.sample_index_list,
                                                            ValidFeaturesLabels.sql_list, valid_table_dict,
                                                            ValidFeaturesLabels.header_question_list, ValidFeaturesLabels.header_table_id_list)
            print(eval_result)

        if do_test:
            test_iterator, TestFeaturesLabels, test_table_dict = self.data_iterator(mode="test")
            print('开始测试')
            sequence_labeling_predict = []
            select_agg_predict = []
            where_relation_predict = []
            where_conlumn_number_predict = []
            type_logits_list = []
            cls_index_list = []
            sel_num_logits_list = []
            where_num_logits_list = []
            type_probs_list = []
            op_logits_list = []
            for j, test_batch_data in enumerate(test_iterator):
                print('testbatchIndex:', j)

                if torch.cuda.is_available():
                    input_ids = test_batch_data[0].to(self.device)
                    attention_masks = test_batch_data[1].to(self.device)
                    cls_indices = test_batch_data[2].to(self.device)
                    header_masks = test_batch_data[3].to(self.device)
                    question_masks = test_batch_data[4].to(self.device)
                    subheader_cls_list = test_batch_data[5].to(self.device)
                    subheader_masks = test_batch_data[6].to(self.device)
                    value_masks = test_batch_data[7].to(self.device)
                    type_masks = test_batch_data[8].to(self.device)
                else:
                    input_ids = test_batch_data[0]
                    attention_masks = test_batch_data[1]
                    cls_indices = test_batch_data[2]
                    header_masks = test_batch_data[3]
                    question_masks = test_batch_data[4]
                    subheader_cls_list = test_batch_data[5]
                    subheader_masks = test_batch_data[6]
                    value_masks = test_batch_data[7]
                    type_masks = test_batch_data[8]
                tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits = model(
                    input_ids, attention_masks, type_masks, header_masks, question_masks, subheader_masks,
                    subheader_cls_list, value_masks, cls_indices)
                sequence_labeling_predict.extend(tag_logits)
                select_agg_predict.extend(agg_logits)
                where_relation_predict.extend(connection_logits)
                where_conlumn_number_predict.extend(con_num_logits)
                type_logits_list.extend(type_logits)
                cls_index_list.extend(cls_indices)
                sel_num_logits_list.extend(sel_num_logits)
                where_num_logits_list.extend(where_num_logits)
                type_probs_list.extend(type_probs)
                op_logits_list.extend(op_logits)

            logits_lists = [sequence_labeling_predict, select_agg_predict, where_relation_predict,
                            where_conlumn_number_predict, type_logits_list, sel_num_logits_list, where_num_logits_list,
                            type_probs_list, op_logits_list]
            labels_lists = [[] for _ in range(8)]
            test_sql_list, test_header_question_list, test_header_table_id_list = [], [], []
            Evaluate.evaluate(ValueOptimizer, logits_lists, cls_index_list, labels_lists, TestFeaturesLabels.question_list, TestFeaturesLabels.question_token_list,
                          TestFeaturesLabels.table_id_list, TestFeaturesLabels.sample_index_list, test_sql_list, test_table_dict,
                          test_header_question_list, test_header_table_id_list, do_test=True)


if __name__ == "__main__":
    TOTAL_START_TIME = time.time()
    data_dir = "./data"
    trainer = Trainer(data_dir, epochs=10, batch_size=32, base_batch_size=32, max_len=128, debug=False)
    # trainer.train(batch_size=16, base_batch_size=16)
    trainer.test(batch_size=64, base_batch_size=64, do_evaluate=False, do_test=True)
    TOTAL_END_TIME = time.time()

    time_cost = TOTAL_END_TIME - TOTAL_START_TIME
    print("It cost {} seconds to finish the test".format(time_cost))
