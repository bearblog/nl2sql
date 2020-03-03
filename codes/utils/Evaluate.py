import os
import copy
import json
import time
import numpy as np
from sklearn.metrics import *
from utils.dbengine import DBEngine
from utils.utils import softmax
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
                 config, logger, do_test=False):
        [
            sequence_labeling_predict,
            select_agg_predict,
            where_relation_predict,
            where_conlumn_number_predict,
            select_where_determine_predict,
            select_number_predict,
            where_number_predict,
            select_where_determine_probs_list,
            where_op_predict
        ] = logits_lists
        select_where_map = {0: 0, 1: 2, 2: 1}

        def swap_type(x):
            x[1], x[2] = x[2], x[1]
            return x
        tmp_list = copy.deepcopy(select_where_determine_probs_list)
        # 这里可能比较难理解，0: select 1: none 2: where
        select_where_determine_predict_new = list(map(lambda x: select_where_map[x], select_where_determine_predict))
        select_where_determine_probs_list_new = list(map(swap_type, tmp_list))

        f_valid = open(os.path.join(config.log_dir,
                                    "badcases{}.txt".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))),
                       'w', encoding="utf-8")
        # {'agg': [0, 0], 'cond_conn_op': 2, 'sel': [1, 3], 'conds': [[0, 2, '6'], [0, 2, '7']]}
        sql_template_forPredict = {"agg": [], "cond_conn_op": None, "sel": [], "conds": []}
        sql_finalPredict = []
        matched_num = 0
        badcase_count = 0
        # 对每一条query进行预测
        for i in range(len(eachData_index)):
            start_index = 0 if i == 0 else eachData_index[i - 1]
            end_index = eachData_index[i]
            eachQuestion = question_list[i]
            eachQuestion_token = question_tokens[i]
            eachtableID = table_id_list[i]
            if do_test is False:
                eachSQL_groundTruth = sql_list_groundTruth[i]
            eachSequence_LabelingPredict = sequence_labeling_predict[start_index: end_index]
            eachSelect_aggPredict = select_agg_predict[start_index: end_index]
            eachWhere_relationPredict = where_relation_predict[start_index: end_index]
            eachWhere_columnNumberPredict = where_conlumn_number_predict[start_index: end_index]
            each_selWhere_deteminePredict = select_where_determine_predict_new[start_index: end_index]
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
                select_where_determine_probs = softmax(select_where_determine_probs_list_new[j])
                select_prob = select_where_determine_probs[0]
                where_prob = select_where_determine_probs[2]
                agg = eachSelect_aggPredict[j]
                select_column = j
                selectProbability_list.append(
                    {"prob": select_prob, "selWhere_detemine": column_type, "sel": select_column, "agg": agg})
                # tag_list是对问题的序列标注 1：代表着是value值
                tag_list = eachSequence_LabelingPredict[j][1: each_firstColumn_CLS_startPosition - 1]
                con_num = eachWhere_columnNumberPredict[j]  # where条件列的数量
                col_op = eachWhere_opPredict[j]
                con_col = j
                candidate_list = [[[], []]]
                candidate_list_index = 0
                value_start_index_list = []
                previous_tag = -1
                question_tag_list = []

                # 去除由于bert_tokenizer分词时产生的##和[UNK]特殊字符的影响，辅助value值的预测 question tokens: ['哪', '几', '年', '的', '商', '品',
                # '房', '新', '开', '工', '面', '积', '超', '过', '了', '9000', '##0', '万', '平', '或', '者', '竣', '工', '面', '积',
                # '高', '于', '50000', '万', '平', '的']
                for i in range(len(tag_list)):
                    tag = tag_list[i]
                    token = eachQuestion_token[i]
                    token = token.replace("##", "")
                    if token == "[UNK]":
                        question_tag_list.extend([tag])
                    else:
                        question_tag_list.extend([tag] * len(token))

                # 得到候选value集合
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

                #  构造候选集  'conds': [[con_col, op, value]]
                if len(value_start_index_list) == 0:
                    con_list = [[con_col, op, ""]]
                else:
                    con_list = []
                for i in range(len(value_start_index_list)):
                    candidate = candidate_list[i]
                    value_start_index = value_start_index_list[i]
                    str_list = candidate[0]
                    # if len(str_list) == 0: continue
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
                            if final_value != value_str:
                                value_Mapping.append([value_str, final_value])
                        elif digit:
                            final_value = digit
                            if final_value != value_str:
                                value_Mapping.append([value_str, final_value])
                    else:
                        if value_str not in col_values:
                            best_value = RegexEngine.select_best_matched_value_from_candidates(
                                candidate_value_set, col_values)
                            if len(best_value) > 0:
                                final_value = best_value
                                if final_value != value_str:
                                    value_Mapping.append([value_str, final_value])
                            else:
                                # value_Mapping.append([value_str, "丢弃"])
                                # continue
                                final_value = value_str
                    con_list.append([con_col, op, final_value])
                    # print("test")
                if len(con_list) == con_num:
                    for con_col, op, final_value in con_list:
                        whereProbability_list.append(
                            {"prob": where_prob, "selWhere_detemine": column_type, "cond": [con_col, op, final_value]})
                else:
                    if len(con_list) > 0:
                        con_col, op, final_value = con_list[0]
                        whereProbability_list.append(
                            {"prob": where_prob, "selWhere_detemine": column_type, "cond": [con_col, op, final_value]})

            sel_num = max(eachSelect_numberPredict, key=eachSelect_numberPredict.count)
            where_num = max(eachWhere_numberPredict, key=eachWhere_numberPredict.count)
            selectProbability_list = sorted(selectProbability_list, key=lambda x: (-x["selWhere_detemine"], x["prob"]),
                                            reverse=True)
            whereProbability_list = sorted(whereProbability_list,
                                           key=lambda x: (x["selWhere_detemine"], x["prob"]),
                                           reverse=True)
            # TODO 预测条件为空值时需要进一步处理
            if where_num == 0:
                print("where num is 0")
                connection = 0
            elif len(whereProbability_list) == 0:
                print("where prob list is empty")
                print(eachQuestion)
                connection = 0
            elif where_num == 1:
                connection = 0
            else:
                # TODO 确定是否有冲突
                where_cols = list(map(lambda x: x["cond"][0], whereProbability_list[: where_num]))
                real_connection_list = [eachWhere_relationPredict[k] for k in where_cols]
                connection = max(real_connection_list, key=real_connection_list.count)

            sql_forPredict = copy.deepcopy(sql_template_forPredict)
            sql_forPredict["cond_conn_op"] = connection
            for j in range(min(sel_num, len(selectProbability_list))):
                sql_forPredict["agg"].append(selectProbability_list[j]["agg"])
                sql_forPredict["sel"].append(selectProbability_list[j]["sel"])
            for j in range(min(where_num, len(whereProbability_list)) if where_num != 0 else 1):
                sql_forPredict["conds"].append(whereProbability_list[j]["cond"])
            sql_finalPredict.append(sql_forPredict)

            if do_test is False:
                if Evaluate.sql_match(sql_forPredict, eachSQL_groundTruth):
                    matched_num += 1
                else:
                    f_valid.write("index: %s\n" % str(badcase_count))
                    f_valid.write("table id: %s\n" % str(eachtableID))
                    f_valid.write("query: %s\n" % str(eachQuestion))
                    f_valid.write("prediction: %s\n" % str(sql_forPredict))
                    f_valid.write("ground truth: %s\n" % str(eachSQL_groundTruth))
                    cols = set(map(lambda x: x[0], sql_forPredict["conds"])) | set(
                        map(lambda x: x[0], eachSQL_groundTruth["conds"])) | set(
                        sql_forPredict["sel"]) | set(eachSQL_groundTruth["sel"])
                    for j, table_header in enumerate(each_conlumnList):
                        if j in cols:
                            f_valid.write("%d、%s\n" % (j, table_header))
                    f_valid.write("\n")
                    badcase_count += 1

        if do_test is False:
            _, _, _ = Evaluate.check_acc(sql_finalPredict, sql_list_groundTruth, table_id_list,
                                         logger, config)

        else:
            # f_result = open(os.path.join(config.submit_dir, "result{}.json".format(time.strftime("%Y-%m-%d", time.localtime(time.time())))),
            #                 'w', encoding="utf-8")
            f_result = open(os.path.join(config.submit_dir, "result.json"),
                            'w', encoding="utf-8")
            for each_sql_predict in sql_finalPredict:
                sql_jsonFile = json.dumps(each_sql_predict, ensure_ascii=False)
                f_result.write(sql_jsonFile + '\n')
            f_result.close()

    @staticmethod
    def check_acc(pred_queries, gt_queries, table_id_list, logger, config):
        tot_err = sel_num_err = agg_err = sel_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = cond_rela_err = 0.0
        total_num = len(pred_queries)
        engine = DBEngine(os.path.join(config.data_dir, "val/val.db"))
        ex_acc_num = 0.0

        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            sel_pred, agg_pred, where_rela_pred = pred_qry['sel'], pred_qry['agg'], pred_qry['cond_conn_op']
            sel_gt, agg_gt, where_rela_gt = gt_qry['sel'], gt_qry['agg'], gt_qry['cond_conn_op']

            if where_rela_gt != where_rela_pred:
                good = False
                cond_rela_err += 1

            if len(sel_pred) != len(sel_gt):
                good = False
                sel_num_err += 1

            pred_sel_dict = {k: v for k, v in zip(list(sel_pred), list(agg_pred))}
            gt_sel_dict = {k: v for k, v in zip(sel_gt, agg_gt)}
            if set(sel_pred) != set(sel_gt):
                good = False
                sel_err += 1
            agg_pred = [pred_sel_dict[x] for x in sorted(pred_sel_dict.keys())]
            agg_gt = [gt_sel_dict[x] for x in sorted(gt_sel_dict.keys())]
            if agg_pred != agg_gt:
                good = False
                agg_err += 1

            cond_pred = pred_qry['conds']
            cond_gt = gt_qry['conds']
            if len(cond_pred) != len(cond_gt):
                good = False
                cond_num_err += 1
            else:
                cond_op_pred, cond_op_gt = {}, {}
                cond_val_pred, cond_val_gt = {}, {}
                for p, g in zip(cond_pred, cond_gt):
                    cond_op_pred[p[0]] = p[1]
                    cond_val_pred[p[0]] = p[2]
                    cond_op_gt[g[0]] = g[1]
                    cond_val_gt[g[0]] = g[2]

                if set(cond_op_pred.keys()) != set(cond_op_gt.keys()):
                    cond_col_err += 1
                    good = False

                where_op_pred = [cond_op_pred[x] for x in sorted(cond_op_pred.keys())]
                where_op_gt = [cond_op_gt[x] for x in sorted(cond_op_gt.keys())]
                if where_op_pred != where_op_gt:
                    cond_op_err += 1
                    good = False

                where_val_pred = [cond_val_pred[x] for x in sorted(cond_val_pred.keys())]
                where_val_gt = [cond_val_gt[x] for x in sorted(cond_val_gt.keys())]
                if where_val_pred != where_val_gt:
                    cond_val_err += 1
                    good = False

            if not good:
                tot_err += 1

        for sql_gt, sql_pred, tid in zip(gt_queries, pred_queries, table_id_list):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
                                          sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)
        logger.info("\nsel_num_acc: %.3f, sel_acc: %.3f, agg_acc: %.3f, cond_num_acc: %.3f, cond_col_acc: %.3f, "
                    "cond_op_acc: %.3f, cond_val_acc: %.3f, "
                    "cond_rela_acc: %.3f" % (
                        1 - (sel_num_err / total_num), 1 - (sel_err / total_num), 1 - (agg_err / total_num),
                        1 - (cond_num_err / total_num),
                        1 - (cond_col_err / total_num), 1 - (cond_op_err / total_num), 1 - (cond_val_err / total_num),
                        1 - (cond_rela_err / total_num)))
        logger.info("Logical Accuracy: %.3f" % (1 - (tot_err / total_num)))
        logger.info("Execution Accuracy: %.3f" % (ex_acc_num / total_num))
        return np.array((sel_num_err, sel_err, agg_err, cond_num_err, cond_col_err, cond_op_err, cond_val_err,
                         cond_rela_err)), tot_err, ex_acc_num
