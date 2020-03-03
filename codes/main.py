import logging
import os
import argparse
import sys
import json
from utils.dbengine import DBEngine

def get_train_logger(args):
    logger = logging.getLogger('train-{}'.format(__name__))
    logger.setLevel(logging.INFO)
    # 控制台
    handler_stream = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler_stream)

    # 日志文件
    handler_file = logging.FileHandler('{}.log'.format(os.path.join(args.dout, args.nick)))
    # formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
    formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    return logger


def get_models():
    return [m.replace('.py', '') for m in os.listdir('models') if not m.startswith('_') and m != 'model']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--epoch', type=int, default=100, help='Epoch number')
    parser.add_argument('--gpu', action='store_true', help='Whether use gpu to train')
    parser.add_argument('--toy', action='store_true', help='If set, use small data for fast debugging')
    parser.add_argument('--ca', action='store_true', help='Whether use column attention')
    parser.add_argument('--train_emb', action='store_true', help='Train word embedding for SQLNet')
    parser.add_argument('--restore', action='store_true', help='Whether restore trained model')
    parser.add_argument('--logdir', type=str, default='log', help='Path of save experiment log')
    parser.add_argument('--mode', type=str, default='train', help='train,adjust,test,rl')
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--emb_size', type=int, default=300, help='embedding size')
    parser.add_argument('--dexp', help='root experiment folder', default='result')
    parser.add_argument('--model', help='which model to use', default='sqlnet', choices=get_models())
    parser.add_argument('-n', '--nick', help='nickname for model', default='sample1000')
    parser.add_argument('--emb_path', help='embedding path', default='char_embedding.json')
    parser.add_argument('--output_dir', type=str, default='predict', help='Output path of prediction result')

    args = parser.parse_args()
    # args.dout = os.path.join(args.dexp,args.model,args.nick)
    args.dout = os.path.join(args.dexp, args.model)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def train(args, use_small):
    pass


def predict(args, use_small):
    logger.info("Output path of prediction result is %s" % args.output_dir)


def read_table(table_path):
    table = {}
    with open(table_path, "r", encoding="utf-8") as table_file:
        for line_index, each_line in enumerate(table_file):
            each_table = json.loads(each_line)
            table[each_table['id']] = each_table
    return table


if __name__ == '__main__':
    engine = DBEngine(os.path.join("../data", "val/val.db"))
    tid = "0de5a3b8351311e9ba1b542696d6e445"
    sql = {'agg': [0], 'cond_conn_op': 2, 'sel': [0], 'conds': [[1, 0, '3677.69'], [4, 0, '-30']]}
    result = engine.execute(tid, sql['sel'], sql['agg'], sql['conds'],
                                          sql['cond_conn_op'])
    tableData = read_table(os.path.join("../data", "val/val.tables.json"))
    print(result)
    print(tableData[tid]["header"])
    for item in tableData[tid]["rows"]:
        print(item)
