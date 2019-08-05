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
from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert.modeling import BertPreTrainedModel



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
        attention_matrix = torch.matmul(col_output_transformed.unsqueeze(1), target_output_transformed.transpose(2, 1)).squeeze(1)
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
                end_index = is_match.regs[0][1] # 最后一个index+1
                if start_index - 1 >= 0 and value[start_index - 1] == "-":
                    start_index -= 1
                longest_num = value[start_index: end_index]
                pre_num, post_num = ValueOptimizer.num_completion(value, question, start_index, end_index, value_start_index, value_end_index, num_type="数字")
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
                pre_num, post_num = ValueOptimizer.num_completion(value, question, start_index, end_index, value_start_index, value_end_index, num_type="中文")
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
        num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
        index_dict = {1: '', 2: '十', 3: '百', 4: '千', 5: '万', 6: '十', 7: '百', 8: '千', 9: '亿'}
        nums = list(num)
        nums_index = [x for x in range(1, len(nums)+1)][-1::-1]
        chinese_num = ''
        for index, item in enumerate(nums):
            chinese_num = "".join((chinese_num, num_dict[item], index_dict[nums_index[index]]))
        chinese_num = re.sub("零[十百千零]*", "零", chinese_num)
        chinese_num = re.sub("零万", "万", chinese_num)
        chinese_num = re.sub("亿万", "亿零", chinese_num)
        chinese_num = re.sub("零零", "零", chinese_num)
        chinese_num = re.sub("零\\b" , "", chinese_num)
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
        num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
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
        if int(num % 1e12) == 0: # 万亿
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
        num_dict = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九', '0': '零', }
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
                    year_list.append("%s年%s月" % (year[-2: ], str(int(month))))
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
                #candidate_substrs.add(tmp_value[index1 + 1: index2])   # 括号不能取出来
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
        if len(digit) > 0 and digit.count(".") <= 1 and QuestionMatcher.is_float(digit) and float(digit) < 1e8 and len(digit) / len(value) > 0.4:
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
                    #print(value, question, sorted_list[0])
                    matched_str = sorted_list[0][0]
        if match_type == "precision":
            precise_substrs = list(filter(lambda x: x[1] == 1, candidate_substrs))
            sorted_list = sorted(precise_substrs, key=lambda x: x[2], reverse=True)
            if len(sorted_list) > 0:
                if sorted_list[0][2] > recall_limit:
                    #print(value, question, sorted_list[0])
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
            candidate_nums = QuestionMatcher.convert_num(value) if "-" not in value else [value]   # - 是负数
            year_list = QuestionMatcher.num2year(value)
            candidate_values = candidate_nums + year_list + [original_value]
        else:
            if value in question:
                candidate_values = [value]
            else:
                candidate_values = QuestionMatcher.convert_str(value)
                if sum([candidate_value in question for candidate_value in candidate_values]) == 0:
                    matched_str = QuestionMatcher.match_str(question, value, precision_limit=0.8, recall_limit=0.65, match_type="recall")
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
            question_3 = "".join(question[matched_index + len(matched_value): ])
            question = question_1 + "[" + question_2 + "]" + question_3
            #print(original_value, question)
        else:
            #print(original_value, "不匹配", question)
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

            tag_logits = torch.argmax(F.log_softmax(tag_output, dim=2), dim=2).detach().cpu().numpy().tolist()
            agg_logits = torch.argmax(F.log_softmax(agg_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            connection_logits = torch.argmax(F.log_softmax(connection_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            con_num_logits = torch.argmax(F.log_softmax(con_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            type_logits = torch.argmax(F.log_softmax(type_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            sel_num_logits = torch.argmax(F.log_softmax(sel_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            where_num_logits = torch.argmax(F.log_softmax(where_num_output, dim=1), dim=1).detach().cpu().numpy().tolist()
            op_logits = torch.argmax(F.log_softmax(op_output, dim=1), dim=1).detach().cpu().numpy().tolist()

            return tag_logits, agg_logits, connection_logits, con_num_logits, type_logits, sel_num_logits, where_num_logits, type_probs, op_logits

