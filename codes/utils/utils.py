import os
import json
# from sqlnet.lib.dbengine import DBEngine
from utils.dbengine import DBEngine
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths,)
    if not isinstance(table_paths, list):
        table_paths = (table_paths,)
    sql_data = []
    table_data = {}

    for SQL_PATH in sql_paths:
        with open(SQL_PATH, encoding='utf-8') as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                if use_small and idx >= 1000:
                    break
                sql_data.append(sql)
        print("Loaded %d data from %s" % (len(sql_data), SQL_PATH))

    for TABLE_PATH in table_paths:
        with open(TABLE_PATH, encoding='utf-8') as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab
        print("Loaded %d data from %s" % (len(table_data), TABLE_PATH))

    # 使得两个数据库sql_data、table_data里的id统一
    ret_sql_data = []
    for sql in sql_data:
        if sql[u'table_id'] in table_data:
            ret_sql_data.append(sql)

    return ret_sql_data, table_data


def load_dataset(path="./data", use_small=False, mode='train'):
    print("Loading dataset")
    train_sql_path = os.path.join(path, "train/train.json")
    train_table_path = os.path.join(path, "train/train.tables.json")
    dev_sql_path = os.path.join(path, "val/val.json")
    dev_table_path = os.path.join(path, "val/val.tables.json")
    test_sql_path = os.path.join(path, "test/test.json")
    test_table_path = os.path.join(path, "test/test.tables.json")

    dev_sql, dev_table = load_data(dev_sql_path, dev_table_path, use_small=use_small)
    dev_db = os.path.join(path, "val/val.db")
    if mode == 'train':
        train_sql, train_table = load_data(train_sql_path, train_table_path, use_small=use_small)
        train_db = os.path.join(path, "train/train.db")
        return train_sql, train_table, train_db, dev_sql, dev_table, dev_db
    elif mode == 'test':
        test_sql, test_table = load_data(test_sql_path, test_table_path, use_small=use_small)
        test_db = os.path.join(path, "test/test.db")
        return dev_sql, dev_table, dev_db, test_sql, test_table, test_db


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    gt_cond_seq = []
    vis_seq = []
    sel_num_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        sel_num = len(sql['sql']['sel'])
        sel_num_seq.append(sel_num)
        conds_num = len(sql['sql']['conds'])
        q_seq.append([char for char in sql['question']])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append(
            (
                len(sql['sql']['agg']),
                sql['sql']['sel'],
                sql['sql']['agg'],
                conds_num,
                tuple(x[0] for x in sql['sql']['conds']),
                tuple(x[1] for x in sql['sql']['conds']),
                sql['sql']['cond_conn_op'],
            ))
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'], table_data[sql['table_id']]['header']))
    if ret_vis_data:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, sel_num_seq, col_seq, col_num, ans_seq, gt_cond_seq


def to_batch_seq_test(sql_data, table_data, idxes, st, ed):
    q_seq = []
    col_seq = []
    col_num = []
    raw_seq = []
    table_ids = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append([char for char in sql['question']])
        col_seq.append([[char for char in header] for header in table_data[sql['table_id']]['header']])
        col_num.append(len(table_data[sql['table_id']]['header']))
        raw_seq.append(sql['question'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return q_seq, col_seq, col_num, raw_seq, table_ids


def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        sql_data[idxes[i]]['sql']['conds'] = sql_data[idxes[i]]['sql']['conds']
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def epoch_train(model, optimizer, batch_size, sql_data, table_data):
    model.train()
    perm = np.random.permutation(len(sql_data))
    perm = list(range(len(sql_data)))
    cum_loss = 0.0
    for st in tqdm(range(len(sql_data) // batch_size + 1)):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq = to_batch_seq(sql_data, table_data, perm, st, ed)
        # q_seq: char-based sequence of question
        # gt_sel_num: number of selected columns and aggregation functions
        # col_seq: char-based column name
        # col_num: number of headers in one table
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
        # gt_cond_seq: ground truth of conds
        gt_where_seq = model.generate_gt_where_seq_test(q_seq, gt_cond_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq,
                              gt_sel_num=gt_sel_num)
        # sel_num_score, sel_col_score, sel_agg_score, cond_score, cond_rela_score

        # compute loss
        loss = model.loss(score, ans_seq, gt_where_seq)
        cum_loss += loss.data.cpu().numpy() * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return cum_loss / len(sql_data)


def predict_test(model, batch_size, sql_data, table_data, output_path):
    model.eval()
    perm = list(range(len(sql_data)))
    fw = open(output_path, 'a', encoding='utf-8')
    for st in tqdm(range(len(sql_data) // batch_size + 1)):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, col_seq, col_num, raw_q_seq, table_ids = to_batch_seq_test(sql_data, table_data, perm, st, ed)
        score = model.forward(q_seq, col_seq, col_num)
        sql_preds = model.gen_query(score, q_seq, col_seq, raw_q_seq)
        for sql_pred in sql_preds:
            # sql_pred = {'sel': [1], 'agg': [0], 'cond_conn_op': 2, 'conds': [[11, 0, '11'], [3, 0, '11']]}
            # fw.writelines(json.dumps(sql_pred,ensure_ascii=False,cls = NumpyEncoder).encode('utf-8')+'\n')
            # fw.writelines(json.dumps(sql_pred,ensure_ascii=False,cls = NumpyEncoder).encode('utf-8'))
            # fw.writelines('\n')
            json.dump(sql_pred, fw, ensure_ascii=False, cls=NumpyEncoder)
            fw.writelines('\n')

    fw.close()


def epoch_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    badcase = 0
    one_acc_num, tot_acc_num, ex_acc_num = 0.0, 0.0, 0.0
    for st in tqdm(range(len(sql_data) // batch_size + 1)):
        ed = (st + 1) * batch_size if (st + 1) * batch_size < len(perm) else len(perm)
        st = st * batch_size
        q_seq, gt_sel_num, col_seq, col_num, ans_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        # q_seq: char-based sequence of question
        # gt_sel_num: number of selected columns and aggregation functions, new added field
        # col_seq: char-based column name
        # col_num: number of headers in one table
        # ans_seq: (sel, number of conds, sel list in conds, op list in conds)
        # gt_cond_seq: ground truth of conditions
        # raw_data: ori question, headers, sql
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        # query_gt: ground truth of sql, data['sql'], containing sel, agg, conds:{sel, op, value}
        raw_q_seq = [x[0] for x in raw_data]  # original question
        try:
            score = model.forward(q_seq, col_seq, col_num)
            pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq)
            # generate predicted format
            one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt)
        except:
            badcase += 1
            print('badcase', badcase)
            continue
        one_acc_num += (ed - st - one_err)
        tot_acc_num += (ed - st - tot_err)

        # Execution Accuracy
        for sql_gt, sql_pred, tid in zip(query_gt, pred_queries, table_ids):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'], sql_gt['cond_conn_op'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'],
                                          sql_pred['cond_conn_op'])
            except:
                ret_pred = None
            ex_acc_num += (ret_gt == ret_pred)
    return one_acc_num / len(sql_data), tot_acc_num / len(sql_data), ex_acc_num / len(sql_data)


def load_word_emb(file_name):
    print('Loading word embedding from %s' % file_name)
    f = open(file_name)
    ret = json.load(f)
    f.close()
    # ret = {}
    # with open(file_name, encoding='latin') as inf:
    #     ret = json.load(inf)
    #     for idx, line in enumerate(inf):
    #         info = line.strip().split(' ')
    #         if info[0].lower() not in ret:
    #             ret[info[0]] = np.array([float(x) for x in info[1:]])
    return ret
