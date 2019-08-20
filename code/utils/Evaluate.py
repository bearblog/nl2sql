import os
import copy
import json
import time
import numpy as np
from sklearn.metrics import *
from modules.regex_engine import RegexEngine
from modules.schema_linking import SchemaLiking


class Evaluate:
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
    def evaluate(logits_lists, firstColumn_CLS_startPositionList, labels_lists, question_list, question_tokens,
                 table_id_list, eachData_index, sql_list_groundTruth, tableData, column_queryList, column_tableidList,
                 config, do_test=False):
        [
            sequence_labeling_predict,
            select_agg_predict,
            where_relation_predict,
            where_conlumn_number_predict,
            sel_where_detemine_predict,
            select_number_predict,
            where_number_predict,
            selWhere_detemine_probs_list,
            where_op_predict
        ] = logits_lists
        [
            sequence_labeling_groudTruth,
            select_agg_groundTruth,
            where_relation_groundTruth,
            where_conlumn_number_groundTruth,
            sel_where_detemine_groundTruth,
            select_number_groundTruth,
            where_number_groundTruth,
            where_op_groundTruth
        ] = labels_lists

        f_valid = open(os.path.join(config.log_dir, "badcases{}.txt".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))),
                       'w', encoding="utf-8")
        # {"agg": [0], "cond_conn_op": 2, "sel": [1], "conds": [[3, 0, "11"], [6, 0, "11"]]}
        sql_template_forPredict = {"agg": [], "cond_conn_op": None, "sel": [], "conds": []}
        sql_finalPredict = []
        matched_num = 0
        for i in range(len(eachData_index)):
            start_index = 0 if i == 0 else eachData_index[i - 1]
            end_index = eachData_index[i]
            eachQuestion = question_list[i]
            eachQuestion_token = question_tokens[i]
            eachtableID = table_id_list[i]
            if do_test is False: eachSQL_groundTruth = sql_list_groundTruth[i]
            eachSequence_LabelingPredict = sequence_labeling_predict[start_index: end_index]
            eachSelect_aggPredict = select_agg_predict[start_index: end_index]
            eachWhere_relationPredict = where_relation_predict[start_index: end_index]
            eachWhere_columnNumberPredict = where_conlumn_number_predict[start_index: end_index]
            each_selWhere_deteminePredict = sel_where_detemine_predict[start_index: end_index]
            eachSelect_numberPredict = select_number_predict[start_index: end_index]
            eachWhere_numberPredict = where_number_predict[start_index: end_index]
            eachWhere_opPredict = where_op_predict[start_index: end_index]

            each_firstColumn_CLS_startPosition = firstColumn_CLS_startPositionList[start_index]
            each_conlumnList = tableData[eachtableID]["header"]
            each_columnTypeList = tableData[eachtableID]["types"]
            each_columnRowList = tableData[eachtableID]["rows"]
            # 维护一个当前预测问题对应的table:each_columnDict
            each_columnDict = {i: [] for i in range(len(each_conlumnList))}
            for row in each_columnRowList:
                for each_column, value in enumerate(row):
                    each_columnDict[each_column].append(str(value))

            # 维护一个模型预测出来的值到矫正过的值的一个哈希映射
            value_Mapping = []
            # 联合概率确定 select 子句
            selectProbability_list = []
            # 联合概率确定 where 子句
            whereProbability_list = []
            for j, column_type in enumerate(each_selWhere_deteminePredict):
                selWhere_detemine_probs = selWhere_detemine_probs_list[j]
                select_prob = selWhere_detemine_probs[0]
                where_prob = selWhere_detemine_probs[1]
                agg = eachSelect_aggPredict[j]
                select_column = j
                selectProbability_list.append(
                    {"prob": select_prob, "selWhere_detemine": column_type, "sel": select_column, "agg": agg})
                tag_list = eachSequence_LabelingPredict[j][1: each_firstColumn_CLS_startPosition - 1]
                con_num = eachWhere_columnNumberPredict[j]
                col_op = eachWhere_opPredict[j]
                con_col = j
                candidate_list = [[[], []]]
                candidate_list_index = 0
                value_start_index_list = []
                previous_tag = -1
                question_tag_list = []
                for i in range(len(tag_list)):
                    tag = tag_list[i]
                    token = eachQuestion_token[i]
                    token = token.replace("##", "")
                    if token == "[UNK]":
                        question_tag_list.extend([tag])
                    else:
                        question_tag_list.extend([tag] * len(token))

                for i in range(0, len(question_tag_list)):
                    current_tag = question_tag_list[i]
                    if current_tag == 0:
                        if previous_tag == 1:
                            candidate_list.append([[], []])
                            candidate_list_index += 1
                    else:
                        if previous_tag in [-1, 0]:
                            value_start_index_list.append(i)
                        candidate_list[candidate_list_index][0].append(eachQuestion[i])  # 多了一个 cls
                        candidate_list[candidate_list_index][1].append(question_tag_list[i])
                    previous_tag = current_tag
                con_list = []
                for i in range(len(value_start_index_list)):
                    candidate = candidate_list[i]
                    value_start_index = value_start_index_list[i]
                    str_list = candidate[0]
                    if len(str_list) == 0: continue
                    value_str = "".join(str_list)
                    header = each_conlumnList[j]
                    col_data_type = each_columnTypeList[j]
                    col_values = each_columnDict[j]
                    op = col_op
                    candidate_value_set = set()
                    new_value, longest_digit_num, longest_chinese_num = RegexEngine.find_longest_num(value_str,
                                                                                                     eachQuestion,
                                                                                                     value_start_index)
                    candidate_value_set.add(value_str)
                    candidate_value_set.add(new_value)
                    if longest_digit_num:
                        candidate_value_set.add(longest_digit_num)
                    digit = None
                    if longest_chinese_num:
                        candidate_value_set.add(longest_chinese_num)
                        digit = RegexEngine.chinese2digits(longest_chinese_num)
                        if digit:
                            candidate_value_set.add(digit)
                    replace_candidate_set = RegexEngine.create_candidate_set(value_str)
                    candidate_value_set |= replace_candidate_set
                    final_value = value_str
                    if op != 2:
                        if longest_digit_num:
                            final_value = longest_digit_num
                            if final_value != value_str: value_Mapping.append([value_str, final_value])
                        elif digit:
                            final_value = digit
                            if final_value != value_str: value_Mapping.append([value_str, final_value])
                    else:
                        if value_str not in col_values:
                            best_value = RegexEngine.select_best_matched_value_from_candidates(
                                candidate_value_set, col_values)
                            if len(best_value) > 0:
                                final_value = best_value
                                if final_value != value_str: value_Mapping.append([value_str, final_value])
                            else:
                                value_Mapping.append([value_str, "丢弃"])
                                continue
                    con_list.append([con_col, op, final_value])
                if len(con_list) == con_num:
                    for [con_col, op, final_value] in con_list:
                        whereProbability_list.append(
                            {"prob": where_prob, "selWhere_detemine": column_type, "cond": [con_col, op, final_value]})
                else:
                    if len(con_list) > 0:
                        [con_col, op, final_value] = con_list[0]
                        whereProbability_list.append(
                            {"prob": where_prob, "selWhere_detemine": column_type, "cond": [con_col, op, final_value]})
            sel_num = max(eachSelect_numberPredict, key=eachSelect_numberPredict.count)
            where_num = max(eachWhere_numberPredict, key=eachWhere_numberPredict.count)
            selectProbability_list = sorted(selectProbability_list, key=lambda x: (-x["selWhere_detemine"], x["prob"]),
                                            reverse=True)
            whereProbability_list = sorted(whereProbability_list,
                                           key=lambda x: (-(x["selWhere_detemine"] ** 2 - 1) ** 2, x["prob"]),
                                           reverse=True)
            if where_num <= 1 or len(whereProbability_list) == 0:
                connection = 0
            else:
                where_cols = list(map(lambda x: x["cond"][0], whereProbability_list[: where_num]))
                real_connection_list = [eachWhere_relationPredict[k] for k in where_cols]
                connection = max(real_connection_list, key=real_connection_list.count)

            sql_forPredict = copy.deepcopy(sql_template_forPredict)
            sql_forPredict["cond_conn_op"] = connection
            for j in range(min(sel_num, len(selectProbability_list))):
                sql_forPredict["agg"].append(selectProbability_list[j]["agg"])
                sql_forPredict["sel"].append(selectProbability_list[j]["sel"])
            for j in range(min(where_num, len(whereProbability_list))):
                sql_forPredict["conds"].append(whereProbability_list[j]["cond"])
            sql_finalPredict.append(sql_forPredict)

            if do_test is False:
                if Evaluate.sql_match(sql_forPredict, eachSQL_groundTruth):
                    matched_num += 1
                else:
                    f_valid.write("%s\n" % str(eachQuestion))
                    f_valid.write("%s\n" % str(sql_forPredict))
                    f_valid.write("%s\n" % str(eachSQL_groundTruth))
                    cols = set(map(lambda x: x[0], sql_forPredict["conds"])) | set(
                        map(lambda x: x[0], eachSQL_groundTruth["conds"]))
                    for j, table_header in enumerate(each_conlumnList):
                        if j in cols:
                            f_valid.write("%d、%s\n" % (j, table_header))
                    f_valid.write("\n")

        if do_test is False:
            logical_acc = matched_num / len(eachData_index)
            print("logical_acc", logical_acc)
            return logical_acc

        else:
            # f_result = open(os.path.join(config.submit_dir, "result{}.json".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))),
            #                 'w', encoding="utf-8")
            f_result = open(os.path.join(config.submit_dir, "result.json"),
                            'w', encoding="utf-8")
            for each_sql_predict in sql_finalPredict:
                sql_jsonFile = json.dumps(each_sql_predict, ensure_ascii=False)
                f_result.write(sql_jsonFile + '\n')
            f_result.close()
