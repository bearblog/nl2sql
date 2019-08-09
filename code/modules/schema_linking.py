import re
import copy
import math
class SchemaLiking:
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
            part1_chinese = SchemaLiking.num2chinese(part1)
            part2_chinese = "".join(list(map(lambda x: num_dict[x], list(part2))))
            chinese_num_set.add(part1_chinese + "点" + part2_chinese)
            chinese_num_set.add(part1_chinese + "块" + part2_chinese)
            chinese_num_set.add(part1_chinese + "元" + part2_chinese)
            if part1 == "0":
                chinese_num_set.add(part2_chinese)
        else:
            chinese_num_set.add(SchemaLiking.num2chinese(num.replace(".", "")))
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
            chinese_num = SchemaLiking.num2chinese(num)
            chinese_nums = SchemaLiking.process_two(chinese_num)
            mix_num = SchemaLiking.create_mix_num(num)
            candidate_nums = SchemaLiking.nums_add_unit(chinese_nums + [num, mix_num])
        else:
            candidate_nums = SchemaLiking.nums_add_unit([num])
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
        if len(digit) > 0 and digit.count(".") <= 1 and SchemaLiking.is_float(digit) and float(digit) < 1e8 and len(digit) / len(value) > 0.4:
            candidate_substrs.add(digit)
            chinese_num_set = SchemaLiking.float2chinese(digit)
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
        if SchemaLiking.is_float(value) and float(value) < 1e8:  # 数字转中文只能支持到亿
            if float(value) - math.floor(float(value)) == 0:
                value = str(int(float(value)))
            candidate_nums = SchemaLiking.convert_num(value) if "-" not in value else [value]   # - 是负数
            year_list = SchemaLiking.num2year(value)
            candidate_values = candidate_nums + year_list + [original_value]
        else:
            if value in question:
                candidate_values = [value]
            else:
                candidate_values = SchemaLiking.convert_str(value)
                if sum([candidate_value in question for candidate_value in candidate_values]) == 0:
                    matched_str = SchemaLiking.match_str(question, value, precision_limit=0.8, recall_limit=0.65, match_type="recall")
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

