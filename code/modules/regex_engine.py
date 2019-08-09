import copy
import re

class RegexEngine:
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
                pre_num, post_num = RegexEngine.num_completion(value, question, start_index, end_index, value_start_index, value_end_index, num_type="数字")
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
                pre_num, post_num = RegexEngine.num_completion(value, question, start_index, end_index, value_start_index, value_end_index, num_type="中文")
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
            value, matched_num = RegexEngine.select_best_matched_value(value, col_values)
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
                digit = RegexEngine._chinese2digits(chinese_num[: index]) + "." + tail
            else:
                digit = RegexEngine._chinese2digits(chinese_num)
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

