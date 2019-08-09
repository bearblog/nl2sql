import torch
# from sqlnet.utils import *
from utils.utils import *
# from sqlnet.model.sqlnet import SQLNet
from models.sqlnet import SQLNet
import logging
import os
import argparse
import sys


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


def main(args):
    if args.mode == 'train':
        if args.toy:
            use_small = True
        else:
            use_small = False
            # load dataset
        train(args, use_small)
        predict(args, use_small)

    if args.mode == 'test':
        if args.toy:
            use_small = True
        else:
            use_small = False
            # load dataset
        predict(args, use_small)

    # elif args.mode == 'adjust':
    #     m.load_model()
    #     m.train()
    # elif args.mode == 'test':
    #     m.load_model()
    #     m.eval()
    # elif args.mode == 'rl':
    #     m.load_model()
    #     m.reinforce_tune()


if __name__ == '__main__':
    args = get_args()
    main(args)
