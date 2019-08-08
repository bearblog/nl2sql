#coding=utf-8

import numpy as np
from sklearn.metrics import *
import copy
import json


class Evaluate:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def detail_score(y_true, y_pred, num_labels, ignore_num=None):
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
        output_str = "overall_acc:%.3f, avg_pre:%.3f, avg_rec:%.3f, avg_f1:%.3f \n" % (
        acc, np.mean(pre_list), np.mean(rec_list), np.mean(f1_list))
        output_str += detail_output_str
        return output_str

    @staticmethod
    def sql_match(s1, s2):
        return (s1['cond_conn_op'] == s2['cond_conn_op']) & \
               (set(zip(s1['sel'], s1['agg'])) == set(zip(s2['sel'], s2['agg']))) & \
               (set([tuple(i) for i in s1['conds']]) == set([tuple(i) for i in s2['conds']]))

    @staticmethod
    def evaluate(optimizer, logits_lists, cls_index_list, labels_lists, question_list, question_token_list, table_id_list,
                 sample_index_list, correct_sql_list, table_dict, header_question_list, header_table_id_list,
                 do_test=False):
        [
            sequence_labeling_predict,
            select_agg_predict,
            where_relation_predict,
            where_conlumn_number_predict,
            type_logits_list,
            sel_num_logits_list,
            where_num_logits_list,
            type_probs_list,
            op_logits_list
        ] = logits_lists
        [
            tag_labels_list,
            agg_labels_list,
            connection_labels_list,
            con_num_labels_list,
            type_labels_list,
            sel_num_labels_list,
            where_num_labels_list,
            op_labels_list
        ] = labels_lists

        # print(tag_logits_list)
        # agg_logits_list,
        # connection_logits_list,
        # con_num_logits_list,
        # type_logits_list,
        # sel_num_logits_list,
        # where_num_logits_list,
        # type_probs_list,
        # op_logits_list
        # exit()
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
            sample_tag_logits = sequence_labeling_predict[start_index: end_index]
            sample_agg_logits = select_agg_predict[start_index: end_index]
            sample_connection_logits = where_relation_predict[start_index: end_index]
            sample_con_num_logits = where_conlumn_number_predict[start_index: end_index]
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

                # 把 token 的 tag_list 扩展成 question 长度
                question_tag_list = []
                for i in range(len(tag_list)):
                    tag = tag_list[i]
                    token = sample_question_token[i]
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
                    header = table_header_list[j]
                    col_data_type = table_type_list[j]
                    col_values = col_dict[j]
                    op = col_op
                    candidate_value_set = set()
                    new_value, longest_digit_num, longest_chinese_num = optimizer.find_longest_num(value_str,
                                                                                                        sample_question,
                                                                                                        value_start_index)
                    candidate_value_set.add(value_str)
                    candidate_value_set.add(new_value)
                    if longest_digit_num:
                        candidate_value_set.add(longest_digit_num)
                    digit = None
                    if longest_chinese_num:
                        candidate_value_set.add(longest_chinese_num)
                        digit = optimizer.chinese2digits(longest_chinese_num)
                        if digit:
                            candidate_value_set.add(digit)
                    replace_candidate_set = optimizer.create_candidate_set(value_str)
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
                            best_value = optimizer.select_best_matched_value_from_candidates(
                                candidate_value_set, col_values)
                            if len(best_value) > 0:
                                final_value = best_value
                                if final_value != value_str: value_change_list.append([value_str, final_value])
                            else:
                                value_change_list.append([value_str, "丢弃"])
                                continue  # =，不在列表内，也没找到模糊匹配，抛弃
                    # con_list 是一列里面的 con
                    con_list.append([con_col, op, final_value])
                if len(con_list) == con_num:
                    for [con_col, op, final_value] in con_list:
                        where_prob_list.append(
                            {"prob": where_prob, "type": col_type, "cond": [con_col, op, final_value]})
                else:
                    if len(con_list) > 0:
                        [con_col, op, final_value] = con_list[0]
                        where_prob_list.append(
                            {"prob": where_prob, "type": col_type, "cond": [con_col, op, final_value]})
            sel_num = max(sample_sel_num_logits, key=sample_sel_num_logits.count)
            where_num = max(sample_where_num_logits, key=sample_where_num_logits.count)

            # connection = max(real_connection_list, key=real_connection_list.count) if where_num > 1 and len(real_connection_list) > 0 else 0
            # type_dict = {0: "sel", 1: "con", 2: "none"}
            sel_prob_list = sorted(sel_prob_list, key=lambda x: (-x["type"], x["prob"]), reverse=True)
            where_prob_list = sorted(where_prob_list, key=lambda x: (-(x["type"] ** 2 - 1) ** 2, x["prob"]),
                                     reverse=True)

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

            if do_test is False:
                if Evaluate.sql_match(tmp_sql_dict, sample_sql):
                    matched_num += 1
                else:
                    f_valid.write("%s\n" % str(sample_question))
                    f_valid.write("%s\n" % str(tmp_sql_dict))
                    f_valid.write("%s\n" % str(sample_sql))
                    # f_valid.write("%s\n" % str(value_change_list))
                    cols = set(map(lambda x: x[0], tmp_sql_dict["conds"])) | set(
                        map(lambda x: x[0], sample_sql["conds"]))
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
                if col_type == 0:  # sel
                    agg_pred.append(select_agg_predict[i])
                    agg_true.append(agg_labels_list[i])
                elif col_type == 1:  # con
                    cls_index = cls_index_list[i]
                    tmp_tag_pred = sequence_labeling_predict[i][1: cls_index - 1]  # 不取 cls 和 sep
                    tmp_tag_true = tag_labels_list[i][1: cls_index - 1]
                    question = header_question_list[i]
                    table_id = header_table_id_list[i]
                    matched = 1 if tmp_tag_pred == tmp_tag_true else 0
                    tag_fully_matched.append(matched)
                    tag_pred.extend(tmp_tag_pred)
                    tag_true.extend(tmp_tag_true)
                    connection_pred.append(where_relation_predict[i])
                    connection_true.append(connection_labels_list[i])
                    con_num_pred.append(where_conlumn_number_predict[i])
                    con_num_true.append(con_num_labels_list[i])
                    op_pred.append(op_logits_list[i])
                    op_true.append(op_labels_list[i])

            eval_result = ""
            eval_result += "TYPE\n" + Evaluate.detail_score(type_true, type_pred, num_labels=3, ignore_num=None) + "\n"
            eval_result += "TAG\n" + Evaluate.detail_score(tag_true, tag_pred, num_labels=2, ignore_num=None) + "\n"
            eval_result += "CONNECTION\n" + Evaluate.detail_score(connection_true, connection_pred, num_labels=3,
                                                              ignore_num=None) + "\n"
            eval_result += "CON_NUM\n" + Evaluate.detail_score(con_num_true, con_num_pred, num_labels=4,
                                                           ignore_num=0) + "\n"
            eval_result += "AGG\n" + Evaluate.detail_score(agg_true, agg_pred, num_labels=6, ignore_num=None) + "\n"
            eval_result += "SEL_NUM\n" + Evaluate.detail_score(sel_num_true, sel_num_pred, num_labels=4,
                                                           ignore_num=0) + "\n"
            eval_result += "WHERE_NUM\n" + Evaluate.detail_score(where_num_true, where_num_pred, num_labels=5,
                                                             ignore_num=0) + "\n"
            eval_result += "OP\n" + Evaluate.detail_score(op_true, op_pred, num_labels=4, ignore_num=None) + "\n"

            tag_acc = accuracy_score(tag_true, tag_pred)

            return eval_result, tag_acc, logical_acc

        else:
            f_result = open("result.json", 'w')
            for sql_dict in sql_list:
                sql_dict_json = json.dumps(sql_dict, ensure_ascii=False)
                f_result.write(sql_dict_json + '\n')
            f_result.close()