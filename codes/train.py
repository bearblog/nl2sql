# -*- coding: utf-8 -*-

import os
from modules.nl2sqlNet import NL2SQL
# from modules.nl2sqlNet_xlnet import NL2SQL
from utils.config import init_logger, timer, model_config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
        self  .value_masks = []
        self.question_token_list = []


class InputFeaturesLabelsForProcess:
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


if __name__ == "__main__":
    config = model_config()

    with timer('Initializing'):
        nl2sql = NL2SQL(config, epochs=4, batch_size=256, step_batch_size=16, max_len=128, debug=True)
    with timer('Training'):
        # nl2sql.train(batch_size=16, step_batch_size=16)
        nl2sql.train()
    # with timer('验证'):
    #     nl2sql.test(batch_size=64, step_batch_size=64, do_evaluate=True, do_test=False)
    # with timer('预测'):
    #     nl2sql.test(batch_size=64, step_batch_size=64, do_evaluate=False, do_test=True)
